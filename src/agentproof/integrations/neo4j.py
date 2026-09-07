"""Neo4j / Memgraph Bolt integration.

`Neo4jIntegration` isolates the official async driver. Memgraph speaks Bolt
and is compatible with this client. Evaluators mock this class so unit tests
never open a network connection.

Usage:
    graph = Neo4jIntegration(config)
    rows = await graph.query(
        "MATCH (n:Entity {email: $email}) RETURN n.id AS id",
        {"email": "claims@acme.com"},
    )
    await graph.close()
"""

import re
from typing import Any

from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import IntegrationError, RetryExhaustedError
from agentproof.core.retry import RetryConfig, retry_async

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_ident(name: str) -> str:
    """Validate and backtick-quote a Cypher identifier (label or rel type).

    Args:
        name: Unquoted identifier.

    Returns:
        Backtick-quoted identifier.

    Raises:
        IntegrationError: If `name` is not a simple identifier.
    """
    if not isinstance(name, str) or not _IDENT.match(name):
        raise IntegrationError(f"invalid Cypher identifier: {name!r}")
    return f"`{name}`"


def record_to_dict(record: Any) -> dict[str, Any]:
    """Normalize a neo4j Record, mapping, or object with `.data()` to a dict.

    Args:
        record: Driver record or mapping.

    Returns:
        Plain dict. Empty dict when `record` is None.
    """
    if record is None:
        return {}
    if isinstance(record, dict):
        return dict(record)
    data = getattr(record, "data", None)
    if callable(data):
        payload = data()
        if isinstance(payload, dict):
            return dict(payload)
    if hasattr(record, "keys"):
        return {str(key): record[key] for key in record.keys()}
    return dict(record)


def extract_node_id(row: dict[str, Any], id_field: str = "id") -> str:
    """Pull a node id from a query row.

    Tries `id_field`, then `node_id`, then `n.id`.

    Args:
        row: Query result mapping.
        id_field: Preferred key.

    Returns:
        String id, or "" if none found.
    """
    for key in (id_field, "node_id", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    nested = row.get("n")
    if isinstance(nested, dict):
        value = nested.get("id") or nested.get("node_id")
        if value is not None:
            return str(value)
    return ""


class Neo4jIntegration:
    """Thin async wrapper around neo4j.AsyncGraphDatabase.

    Driver construction does not open a socket; the first `query` does.
    Inject `driver` in unit tests to skip the SDK entirely.

    Args:
        config: AgentProofConfig with `neo4j_url`, user, password, database.
        driver: Optional pre-built async driver (or test double).
    """

    def __init__(
        self,
        config: AgentProofConfig,
        driver: Any | None = None,
    ) -> None:
        self.config = config
        if driver is not None:
            self._driver = driver
        else:
            auth = (config.neo4j_user, config.neo4j_password or "")
            self._driver = AsyncGraphDatabase.driver(config.neo4j_url, auth=auth)
        self._retry = RetryConfig(
            max_attempts=config.max_retries,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
            retryable_exceptions=(
                ConnectionError,
                TimeoutError,
                OSError,
                ServiceUnavailable,
                SessionExpired,
                TransientError,
            ),
        )

    async def query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a Cypher query and return rows as dicts.

        Args:
            cypher: Parameterized Cypher (`$name` placeholders).
            parameters: Bind parameters.

        Returns:
            List of column-name mappings.

        Raises:
            IntegrationError: Empty Cypher, or the call failed after retries.
        """
        self._require_cypher(cypher)
        params = parameters or {}

        async def _call() -> list[dict[str, Any]]:
            async with self._driver.session(database=self.config.neo4j_database) as session:
                result = await session.run(cypher, params)
                rows: list[dict[str, Any]] = []
                async for record in result:
                    rows.append(record_to_dict(record))
                return rows

        return await self._run("query", _call)

    async def execute(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> int:
        """Run a write query and return the consumed counter if available.

        Args:
            cypher: Parameterized Cypher.
            parameters: Bind parameters.

        Returns:
            `nodes_created + relationships_created` from the summary when
            present, otherwise 0.

        Raises:
            IntegrationError: Empty Cypher, or the call failed.
        """
        self._require_cypher(cypher)
        params = parameters or {}

        async def _call() -> int:
            async with self._driver.session(database=self.config.neo4j_database) as session:
                result = await session.run(cypher, params)
                summary = await result.consume()
            counters = getattr(summary, "counters", None)
            if counters is None:
                return 0
            nodes = int(getattr(counters, "nodes_created", 0) or 0)
            rels = int(getattr(counters, "relationships_created", 0) or 0)
            return nodes + rels

        return await self._run("execute", _call)

    async def close(self) -> None:
        """Close the underlying driver if it exposes close()."""
        close = getattr(self._driver, "close", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            await result

    async def _run(self, operation: str, call: Any) -> Any:
        """Execute `call` with retry; map failures to IntegrationError.

        Args:
            operation: Short name used in error messages.
            call: Zero-arg async callable.

        Returns:
            The callable's return value.

        Raises:
            IntegrationError: Retries exhausted or a non-retryable error.
        """
        try:
            return await retry_async(self._retry)(call)()
        except RetryExhaustedError as exc:
            raise IntegrationError(
                f"Neo4j {operation} failed after {self._retry.max_attempts} "
                f"attempts: {exc.last_exception}"
            ) from exc
        except IntegrationError:
            raise
        except Exception as exc:
            raise IntegrationError(f"Neo4j {operation} failed: {exc}") from exc

    @staticmethod
    def _require_cypher(cypher: str) -> None:
        """Raise IntegrationError when `cypher` is empty or whitespace."""
        if not isinstance(cypher, str) or not cypher.strip():
            raise IntegrationError("cypher must be a non-empty string")
