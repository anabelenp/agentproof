"""Nango connector HTTP integration.

`NangoIntegration` isolates the Nango REST API (records, sync trigger,
connection/auth). Evaluators inject an httpx client so unit tests never
open a network connection.

Usage:
    nango = NangoIntegration(config)
    records = await nango.list_records(connection_id="salesforce-1", model="Contact")
    await nango.close()
"""

from typing import Any

import httpx

from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import IntegrationError, RetryExhaustedError
from agentproof.core.retry import RetryConfig, retry_async

RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class NangoTransientError(Exception):
    """Retryable Nango HTTP failure (429 / 5xx)."""


def normalize_records(payload: Any) -> list[dict[str, Any]]:
    """Extract a list of record dicts from a Nango-style payload.

    Args:
        payload: JSON body — a list, or a mapping with `records` / `data`.

    Returns:
        List of dict rows. Non-dict items are skipped.
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("records")
        if rows is None:
            rows = payload.get("data")
        if rows is None:
            return [payload] if payload else []
    else:
        return []
    return [dict(item) for item in rows if isinstance(item, dict)]


class NangoIntegration:
    """Thin async wrapper around the Nango REST API via httpx.

    Args:
        config: AgentProofConfig. `nango_base_url` and `nango_api_key`
            select the server. A missing key is allowed when a client is
            injected (unit tests).
        client: Optional httpx.AsyncClient (or test double).
    """

    def __init__(
        self,
        config: AgentProofConfig,
        client: httpx.AsyncClient | Any | None = None,
    ) -> None:
        self.config = config
        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            headers: dict[str, str] = {}
            if config.nango_api_key:
                headers["Authorization"] = f"Bearer {config.nango_api_key}"
            self._client = httpx.AsyncClient(
                base_url=config.nango_base_url.rstrip("/"),
                headers=headers,
                timeout=30.0,
            )
        self._retry = RetryConfig(
            max_attempts=config.max_retries,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
            retryable_exceptions=(
                ConnectionError,
                TimeoutError,
                OSError,
                httpx.TransportError,
                httpx.TimeoutException,
                NangoTransientError,
            ),
        )

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        """Send an HTTP request and return parsed JSON.

        Args:
            method: HTTP method.
            path: Path relative to `nango_base_url`.
            params: Query string.
            json: JSON body.

        Returns:
            Parsed JSON (dict, list, or {}).

        Raises:
            IntegrationError: Non-retryable HTTP error or retries exhausted.
        """
        if not path or not str(path).strip():
            raise IntegrationError("path must be a non-empty string")

        async def _call() -> Any:
            response = await self._client.request(
                method.upper(),
                path,
                params=params,
                json=json,
            )
            status = int(getattr(response, "status_code", 0) or 0)
            if status in RETRYABLE_STATUSES:
                raise NangoTransientError(
                    f"Nango {method.upper()} {path} returned {status}"
                )
            if status >= 400:
                raise IntegrationError(
                    f"Nango {method.upper()} {path} returned {status}"
                )
            content = getattr(response, "content", None)
            if not content:
                return {}
            json_fn = getattr(response, "json", None)
            if callable(json_fn):
                payload = json_fn()
                if hasattr(payload, "__await__"):
                    payload = await payload
                return payload
            return {}

        try:
            return await retry_async(self._retry)(_call)()
        except RetryExhaustedError as exc:
            raise IntegrationError(
                f"Nango {method.upper()} {path} failed after "
                f"{self._retry.max_attempts} attempts: {exc.last_exception}"
            ) from exc
        except IntegrationError:
            raise
        except Exception as exc:
            raise IntegrationError(f"Nango {method.upper()} {path} failed: {exc}") from exc

    async def list_records(
        self,
        *,
        connection_id: str,
        model: str,
        provider_config_key: str = "",
    ) -> list[dict[str, Any]]:
        """Fetch synced records for a connection and model.

        Args:
            connection_id: Nango connection id.
            model: Synced model name (e.g. `Contact`).
            provider_config_key: Optional integration unique key.

        Returns:
            Record dicts.

        Raises:
            IntegrationError: Missing arguments or the call failed.
        """
        if not connection_id or not connection_id.strip():
            raise IntegrationError("connection_id must be a non-empty string")
        if not model or not model.strip():
            raise IntegrationError("model must be a non-empty string")
        params: dict[str, Any] = {
            "connectionId": connection_id,
            "model": model,
        }
        if provider_config_key:
            params["providerConfigKey"] = provider_config_key
        payload = await self.request("GET", "/records", params=params)
        return normalize_records(payload)

    async def trigger_sync(
        self,
        *,
        connection_id: str,
        provider_config_key: str = "",
        syncs: list[str] | None = None,
    ) -> dict[str, Any]:
        """Trigger a Nango sync for a connection.

        Args:
            connection_id: Nango connection id.
            provider_config_key: Optional integration unique key.
            syncs: Optional sync names to run.

        Returns:
            JSON body from Nango.

        Raises:
            IntegrationError: Missing connection_id or the call failed.
        """
        if not connection_id or not connection_id.strip():
            raise IntegrationError("connection_id must be a non-empty string")
        body: dict[str, Any] = {"connection_id": connection_id}
        if provider_config_key:
            body["provider_config_key"] = provider_config_key
        if syncs:
            body["syncs"] = list(syncs)
        payload = await self.request("POST", "/sync/trigger", json=body)
        return payload if isinstance(payload, dict) else {"data": payload}

    async def get_connection(
        self,
        connection_id: str,
        *,
        provider_config_key: str = "",
    ) -> dict[str, Any]:
        """Fetch connection metadata (auth status, credentials expiry).

        Args:
            connection_id: Nango connection id.
            provider_config_key: Optional integration unique key.

        Returns:
            Connection JSON.

        Raises:
            IntegrationError: Missing connection_id or the call failed.
        """
        if not connection_id or not connection_id.strip():
            raise IntegrationError("connection_id must be a non-empty string")
        params: dict[str, Any] = {}
        if provider_config_key:
            params["provider_config_key"] = provider_config_key
        payload = await self.request(
            "GET", f"/connection/{connection_id}", params=params or None
        )
        return payload if isinstance(payload, dict) else {}

    async def close(self) -> None:
        """Close the underlying HTTP client when this instance owns it."""
        if not self._owns_client:
            return
        aclose = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if aclose is None:
            return
        result = aclose()
        if hasattr(result, "__await__"):
            await result
