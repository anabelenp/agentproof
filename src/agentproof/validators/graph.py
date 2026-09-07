"""GraphValidator — Neo4j / Memgraph knowledge-graph integrity.

Scores entity resolution, relationship type/direction, temporal event order,
known-query correctness, and whether injected anomalies are surfaced.

Unit tests pass in-memory rows (`resolved_nodes`, `relationships`, `events`,
`rows`, `detected`) so no Bolt connection is required. Live checks go through
`Neo4jIntegration.query`.

Usage:
    validator = GraphValidator(config, audit_logger, integration=neo4j)
    result = await validator.validate_entity_resolution(
        entity_sources=[
            {"entity_key": "acme", "email": "a@acme.com", "source": "crm"},
            {"entity_key": "acme", "email": "acme@mail.com", "source": "email"},
        ],
        expected_node_count=1,
    )
"""

from datetime import datetime, timezone
from typing import Any

from agentproof.core.audit import AuditLogger
from agentproof.core.base import BaseEvaluator, ValidationResult
from agentproof.core.config import AgentProofConfig
from agentproof.core.errors import AuditError, GraphValidatorError
from agentproof.integrations.neo4j import (
    Neo4jIntegration,
    extract_node_id,
    quote_ident,
)

SUPPORTED_METRICS = (
    "entity_resolution",
    "relationship_integrity",
    "temporal_accuracy",
    "query_correctness",
    "failure_modes",
)

_DIRECTION_ALIASES = {
    "outgoing": "outgoing",
    "out": "outgoing",
    "->": "outgoing",
    "forward": "outgoing",
    "incoming": "incoming",
    "in": "incoming",
    "<-": "incoming",
    "backward": "incoming",
    "both": "both",
    "<->": "both",
}


def normalize_direction(value: str) -> str:
    """Map direction spellings to outgoing / incoming / both.

    Args:
        value: Caller or graph direction string.

    Returns:
        Canonical direction. Unknown values are lowercased as-is.
    """
    key = value.strip().lower()
    return _DIRECTION_ALIASES.get(key, key)


def group_sources(entity_sources: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group source records by `entity_key`, or one group if omitted.

    Args:
        entity_sources: Records that should collapse to graph nodes.

    Returns:
        Mapping of entity key to source dicts.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in entity_sources:
        key = str(source.get("entity_key") or source.get("entity") or "_default")
        grouped.setdefault(key, []).append(source)
    return grouped


def events_in_order(actual: list[str], expected: list[str]) -> bool:
    """True if `expected` is an ordered subsequence of `actual`.

    Args:
        actual: Event names in timestamp order (may include extras).
        expected: Required sequence.

    Returns:
        True if every expected name appears in order.
    """
    iterator = iter(actual)
    for name in expected:
        for observed in iterator:
            if observed == name:
                break
        else:
            return False
    return True


def _clamp(score: float) -> float:
    """Clamp a score into ValidationResult's legal range [0.0, 1.0]."""
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _anomaly_key(item: dict[str, Any]) -> tuple[str, str]:
    """Stable (kind, id) pair for comparing injected vs detected anomalies."""
    kind = str(item.get("kind") or item.get("type") or "").strip().lower()
    ident = str(
        item.get("id")
        or item.get("entity_id")
        or item.get("node_id")
        or item.get("name")
        or ""
    )
    return kind, ident


class GraphValidator(BaseEvaluator):
    """Scores knowledge-graph integrity for Neo4j and Memgraph.

    Evaluation failures never raise — they return a ValidationResult with
    `error` set. AuditError is the only exception that propagates.

    Args:
        config: AgentProofConfig with neo4j URL and entity-resolution threshold.
        audit_logger: AuditLogger that receives every result.
        integration: Optional Neo4jIntegration. Created from config if omitted.
    """

    def __init__(
        self,
        config: AgentProofConfig,
        audit_logger: AuditLogger,
        integration: Neo4jIntegration | None = None,
    ) -> None:
        super().__init__(config, audit_logger)
        self._integration = integration or Neo4jIntegration(config)

    @property
    def name(self) -> str:
        """Unique evaluator identifier used in audit logs and reports."""
        return "GraphValidator"

    @property
    def integration(self) -> Neo4jIntegration:
        """The Neo4jIntegration used for Cypher queries."""
        return self._integration

    async def evaluate(
        self,
        *,
        metric: str = "entity_resolution",
        entity_sources: list[dict[str, Any]] | None = None,
        expected_node_count: int = 1,
        resolved_nodes: list[dict[str, Any]] | None = None,
        label: str = "Entity",
        match_keys: list[str] | None = None,
        cypher_query: str = "",
        expected_relationship_type: str = "",
        expected_direction: str = "outgoing",
        parameters: dict[str, Any] | None = None,
        relationships: list[dict[str, Any]] | None = None,
        entity_id: str = "",
        expected_event_sequence: list[str] | None = None,
        events: list[dict[str, Any]] | None = None,
        expected_node_ids: list[str] | None = None,
        rows: list[dict[str, Any]] | None = None,
        id_field: str = "id",
        injected_anomalies: list[dict[str, Any]] | None = None,
        detected: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Dispatch to a graph integrity check.

        Args:
            metric: entity_resolution (default), relationship_integrity,
                temporal_accuracy, query_correctness, failure_modes.
            entity_sources: Source records for entity_resolution.
            expected_node_count: Nodes each entity should collapse to.
            resolved_nodes: In-memory matches; skips Bolt when set.
            label / match_keys: Live MATCH helpers.
            cypher_query: Query for relationship_integrity / query_correctness.
            expected_relationship_type / expected_direction: Edge contract.
            parameters: Cypher bind parameters.
            relationships: In-memory edges; skips Bolt when set.
            entity_id / expected_event_sequence / events: Temporal check.
            expected_node_ids / rows / id_field: Query-correctness check.
            injected_anomalies / detected: Failure-mode check.

        Returns:
            ValidationResult for the requested metric.
        """
        key = metric.lower().strip()
        if key == "entity_resolution":
            return await self.validate_entity_resolution(
                entity_sources=entity_sources or [],
                expected_node_count=expected_node_count,
                resolved_nodes=resolved_nodes,
                label=label,
                match_keys=match_keys,
            )
        if key == "relationship_integrity":
            return await self.validate_relationship_integrity(
                cypher_query=cypher_query,
                expected_relationship_type=expected_relationship_type,
                expected_direction=expected_direction,
                parameters=parameters,
                relationships=relationships,
            )
        if key == "temporal_accuracy":
            return await self.validate_temporal_accuracy(
                entity_id=entity_id,
                expected_event_sequence=expected_event_sequence or [],
                events=events,
            )
        if key == "query_correctness":
            return await self.validate_query_correctness(
                cypher_query=cypher_query,
                expected_node_ids=expected_node_ids or [],
                parameters=parameters,
                rows=rows,
                id_field=id_field,
            )
        if key == "failure_modes":
            return await self.validate_failure_modes(
                injected_anomalies=injected_anomalies or [],
                detected=detected,
            )

        audit_id = self._new_audit_id()
        start = self._start_timer()
        result = ValidationResult(
            passed=False,
            score=0.0,
            evaluator_name=self.name,
            metric=metric,
            threshold=0.0,
            details={"supported_metrics": list(SUPPORTED_METRICS)},
            latency_ms=self._elapsed_ms(start),
            timestamp=datetime.now(timezone.utc),
            audit_id=audit_id,
            error=(
                f"Unknown metric {metric!r}. "
                f"Expected one of: {', '.join(SUPPORTED_METRICS)}"
            ),
        )
        await self._write_audit(result)
        return result

    async def validate_entity_resolution(
        self,
        *,
        entity_sources: list[dict[str, Any]],
        expected_node_count: int = 1,
        resolved_nodes: list[dict[str, Any]] | None = None,
        label: str = "Entity",
        match_keys: list[str] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm source records for each entity collapse to one graph node.

        Args:
            entity_sources: Records from CRM/email/calendar. Optional
                `entity_key` groups records that must share a node.
            expected_node_count: Unique nodes per group (default 1).
            resolved_nodes: In-memory `{entity_key, node_id}` rows. Loaded
                from Neo4j when omitted.
            label: Node label for live MATCH.
            match_keys: Properties to OR-match. Defaults to all keys except
                `entity_key` / `entity` / `source`.

        Returns:
            ValidationResult. score is groups-at-expected-count / groups.
            passed when score >= `entity_resolution_threshold` (0.95).
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        threshold = self.config.entity_resolution_threshold
        try:
            if not entity_sources:
                raise GraphValidatorError(
                    "entity_resolution requires non-empty entity_sources",
                    evaluator_name=self.name,
                    metric="entity_resolution",
                )
            if expected_node_count < 1:
                raise GraphValidatorError(
                    "expected_node_count must be >= 1",
                    evaluator_name=self.name,
                    metric="entity_resolution",
                )
            grouped = group_sources(entity_sources)
            matches = resolved_nodes
            if matches is None:
                matches = await self._load_resolved_nodes(
                    grouped, label=label, match_keys=match_keys
                )

            by_key: dict[str, set[str]] = {key: set() for key in grouped}
            for row in matches:
                key = str(row.get("entity_key") or row.get("entity") or "_default")
                node_id = extract_node_id(row)
                if key not in by_key:
                    by_key[key] = set()
                if node_id:
                    by_key[key].add(node_id)

            over_merged: list[str] = []
            split: list[str] = []
            missing: list[str] = []
            ok = 0
            for key in grouped:
                count = len(by_key.get(key, set()))
                if count == expected_node_count:
                    ok += 1
                elif count == 0:
                    missing.append(key)
                elif count < expected_node_count:
                    split.append(key)
                else:
                    over_merged.append(key)

            total = len(grouped)
            score = _clamp(ok / total) if total else 1.0
            result = ValidationResult(
                passed=score >= threshold,
                score=score,
                evaluator_name=self.name,
                metric="entity_resolution",
                threshold=threshold,
                details={
                    "group_count": total,
                    "groups_ok": ok,
                    "expected_node_count": expected_node_count,
                    "missing_entities": missing,
                    "split_entities": split,
                    "over_merged_entities": over_merged,
                    "node_ids": {key: sorted(ids) for key, ids in by_key.items()},
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "entity_resolution", exc, threshold=threshold
            )

        await self._write_audit(result)
        return result

    async def validate_relationship_integrity(
        self,
        *,
        cypher_query: str = "",
        expected_relationship_type: str,
        expected_direction: str = "outgoing",
        parameters: dict[str, Any] | None = None,
        relationships: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm edges match the expected type and direction.

        Zero-tolerance: any mistyped or misdirected edge fails.

        Args:
            cypher_query: Query returning `type` and `direction` (or start/end).
            expected_relationship_type: Rel type (e.g. `DECIDED`).
            expected_direction: outgoing, incoming, or both (`->` / `<-` ok).
            parameters: Cypher bind parameters.
            relationships: In-memory edges; skips Bolt when set.

        Returns:
            ValidationResult. score is matching-edges / total-edges.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            self._require_text(
                expected_relationship_type, "expected_relationship_type", "relationship_integrity"
            )
            want_type = expected_relationship_type.strip()
            want_dir = normalize_direction(expected_direction)
            if relationships is None:
                if not cypher_query.strip():
                    raise GraphValidatorError(
                        "relationship_integrity requires cypher_query or relationships",
                        evaluator_name=self.name,
                        metric="relationship_integrity",
                    )
                relationships = await self._integration.query(cypher_query, parameters)

            if not relationships:
                result = ValidationResult(
                    passed=False,
                    score=0.0,
                    evaluator_name=self.name,
                    metric="relationship_integrity",
                    threshold=1.0,
                    details={
                        "expected_type": want_type,
                        "expected_direction": want_dir,
                        "empty": True,
                        "mismatched": [],
                    },
                    latency_ms=self._elapsed_ms(start),
                    timestamp=datetime.now(timezone.utc),
                    audit_id=audit_id,
                )
            else:
                mismatched: list[dict[str, str]] = []
                matching = 0
                for edge in relationships:
                    actual_type = str(
                        edge.get("type") or edge.get("rel_type") or edge.get("relationship") or ""
                    )
                    actual_dir = normalize_direction(
                        str(edge.get("direction") or _infer_direction(edge) or "")
                    )
                    type_ok = actual_type.lower() == want_type.lower()
                    dir_ok = want_dir == "both" or actual_dir == want_dir
                    if type_ok and dir_ok:
                        matching += 1
                    else:
                        mismatched.append(
                            {"type": actual_type, "direction": actual_dir}
                        )
                score = _clamp(matching / len(relationships))
                result = ValidationResult(
                    passed=not mismatched,
                    score=score,
                    evaluator_name=self.name,
                    metric="relationship_integrity",
                    threshold=1.0,
                    details={
                        "expected_type": want_type,
                        "expected_direction": want_dir,
                        "edge_count": len(relationships),
                        "matching": matching,
                        "mismatched": mismatched,
                        "empty": False,
                    },
                    latency_ms=self._elapsed_ms(start),
                    timestamp=datetime.now(timezone.utc),
                    audit_id=audit_id,
                )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "relationship_integrity", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_temporal_accuracy(
        self,
        *,
        entity_id: str,
        expected_event_sequence: list[str],
        events: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm events for `entity_id` appear in the expected time order.

        Args:
            entity_id: Node whose timeline is checked.
            expected_event_sequence: Ordered event names.
            events: In-memory `{name, timestamp}` rows. Loaded from Neo4j
                when omitted.

        Returns:
            ValidationResult. passed only when every expected event is present
            in order. score is found/expected.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            self._require_text(entity_id, "entity_id", "temporal_accuracy")
            if not expected_event_sequence:
                raise GraphValidatorError(
                    "temporal_accuracy requires a non-empty expected_event_sequence",
                    evaluator_name=self.name,
                    metric="temporal_accuracy",
                )
            if events is None:
                events = await self._integration.query(
                    "MATCH (e {id: $id})-[:HAS_EVENT]->(ev) "
                    "RETURN ev.name AS name, ev.timestamp AS timestamp "
                    "ORDER BY ev.timestamp",
                    {"id": entity_id},
                )
            ordered = _sort_events(events)
            names = [
                str(item.get("name") or item.get("event") or item.get("type") or "")
                for item in ordered
            ]
            names = [name for name in names if name]
            missing = [name for name in expected_event_sequence if name not in names]
            order_ok = events_in_order(names, expected_event_sequence)
            found = len(expected_event_sequence) - len(missing)
            score = _clamp(found / len(expected_event_sequence))
            result = ValidationResult(
                passed=not missing and order_ok,
                score=score,
                evaluator_name=self.name,
                metric="temporal_accuracy",
                threshold=1.0,
                details={
                    "entity_id": entity_id,
                    "expected_sequence": list(expected_event_sequence),
                    "actual_sequence": names,
                    "missing": missing,
                    "order_ok": order_ok,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "temporal_accuracy", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_query_correctness(
        self,
        *,
        cypher_query: str,
        expected_node_ids: list[str],
        parameters: dict[str, Any] | None = None,
        rows: list[dict[str, Any]] | None = None,
        id_field: str = "id",
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm a known Cypher query returns the expected node ids.

        Args:
            cypher_query: Query against a known dataset.
            expected_node_ids: Ids that must be returned (exact set).
            parameters: Cypher bind parameters.
            rows: In-memory result rows; skips Bolt when set.
            id_field: Column holding the node id.

        Returns:
            ValidationResult. score is F1 of retrieved vs expected ids.
            passed only when the sets are equal.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            if not expected_node_ids:
                raise GraphValidatorError(
                    "query_correctness requires non-empty expected_node_ids",
                    evaluator_name=self.name,
                    metric="query_correctness",
                )
            if rows is None:
                if not cypher_query.strip():
                    raise GraphValidatorError(
                        "query_correctness requires cypher_query or rows",
                        evaluator_name=self.name,
                        metric="query_correctness",
                    )
                rows = await self._integration.query(cypher_query, parameters)
            got = [extract_node_id(row, id_field) for row in rows]
            got_ids = [item for item in got if item]
            expected = [str(item) for item in expected_node_ids]
            got_set = set(got_ids)
            exp_set = set(expected)
            hits = got_set & exp_set
            precision = len(hits) / len(got_set) if got_set else 0.0
            recall = len(hits) / len(exp_set) if exp_set else 0.0
            if precision + recall == 0.0:
                score = 0.0
            else:
                score = _clamp(2.0 * precision * recall / (precision + recall))
            result = ValidationResult(
                passed=got_set == exp_set,
                score=score,
                evaluator_name=self.name,
                metric="query_correctness",
                threshold=1.0,
                details={
                    "expected_ids": expected,
                    "returned_ids": got_ids,
                    "missing_ids": sorted(exp_set - got_set),
                    "extra_ids": sorted(got_set - exp_set),
                    "precision": precision,
                    "recall": recall,
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "query_correctness", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def validate_failure_modes(
        self,
        *,
        injected_anomalies: list[dict[str, Any]],
        detected: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Confirm injected graph anomalies are surfaced, not silent.

        Args:
            injected_anomalies: `{kind, id}` records (missing, duplicate,
                contradictory, malformed).
            detected: Anomalies the graph layer reported. When omitted the
                integration is queried for each injected id.

        Returns:
            ValidationResult. score is surfaced / injected. passed only when
            every injected anomaly is detected.
        """
        audit_id = self._new_audit_id()
        start = self._start_timer()
        try:
            if not injected_anomalies:
                raise GraphValidatorError(
                    "failure_modes requires non-empty injected_anomalies",
                    evaluator_name=self.name,
                    metric="failure_modes",
                )
            reported = detected
            if reported is None:
                reported = await self._probe_anomalies(injected_anomalies)
            injected_keys = [_anomaly_key(item) for item in injected_anomalies]
            detected_keys = {_anomaly_key(item) for item in reported}
            surfaced = [key for key in injected_keys if key in detected_keys]
            silent = [key for key in injected_keys if key not in detected_keys]
            score = _clamp(len(surfaced) / len(injected_keys))
            result = ValidationResult(
                passed=not silent,
                score=score,
                evaluator_name=self.name,
                metric="failure_modes",
                threshold=1.0,
                details={
                    "injected": [list(key) for key in injected_keys],
                    "surfaced": [list(key) for key in surfaced],
                    "silent": [list(key) for key in silent],
                },
                latency_ms=self._elapsed_ms(start),
                timestamp=datetime.now(timezone.utc),
                audit_id=audit_id,
            )
        except AuditError:
            raise
        except Exception as exc:
            result = self._error_result(
                audit_id, start, "failure_modes", exc, threshold=1.0
            )

        await self._write_audit(result)
        return result

    async def evaluate_all(
        self,
        *,
        entity_sources: list[dict[str, Any]] | None = None,
        expected_node_count: int = 1,
        resolved_nodes: list[dict[str, Any]] | None = None,
        cypher_query: str = "",
        expected_relationship_type: str = "",
        expected_direction: str = "outgoing",
        relationships: list[dict[str, Any]] | None = None,
        entity_id: str = "",
        expected_event_sequence: list[str] | None = None,
        events: list[dict[str, Any]] | None = None,
        expected_node_ids: list[str] | None = None,
        rows: list[dict[str, Any]] | None = None,
        injected_anomalies: list[dict[str, Any]] | None = None,
        detected: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[ValidationResult]:
        """Run every graph check that has enough arguments.

        Returns:
            ValidationResult list. Individual failures do not abort the rest.
        """
        results: list[ValidationResult] = []
        if entity_sources:
            results.append(
                await self.validate_entity_resolution(
                    entity_sources=entity_sources,
                    expected_node_count=expected_node_count,
                    resolved_nodes=resolved_nodes,
                )
            )
        if expected_relationship_type and (cypher_query or relationships is not None):
            results.append(
                await self.validate_relationship_integrity(
                    cypher_query=cypher_query,
                    expected_relationship_type=expected_relationship_type,
                    expected_direction=expected_direction,
                    relationships=relationships,
                )
            )
        if entity_id and expected_event_sequence:
            results.append(
                await self.validate_temporal_accuracy(
                    entity_id=entity_id,
                    expected_event_sequence=expected_event_sequence,
                    events=events,
                )
            )
        if expected_node_ids and (cypher_query or rows is not None):
            results.append(
                await self.validate_query_correctness(
                    cypher_query=cypher_query,
                    expected_node_ids=expected_node_ids,
                    rows=rows,
                )
            )
        if injected_anomalies:
            results.append(
                await self.validate_failure_modes(
                    injected_anomalies=injected_anomalies,
                    detected=detected,
                )
            )
        return results

    async def _load_resolved_nodes(
        self,
        grouped: dict[str, list[dict[str, Any]]],
        *,
        label: str,
        match_keys: list[str] | None,
    ) -> list[dict[str, Any]]:
        """MATCH nodes whose properties overlap each source group.

        Args:
            grouped: entity_key → source records.
            label: Node label.
            match_keys: Properties to compare.

        Returns:
            Rows with entity_key and node_id.
        """
        quoted = quote_ident(label)
        skip = {"entity_key", "entity", "source"}
        found: list[dict[str, Any]] = []
        for key, sources in grouped.items():
            keys = match_keys or [
                name
                for source in sources
                for name in source
                if name not in skip and source[name] not in (None, "")
            ]
            keys = list(dict.fromkeys(keys))
            values: list[Any] = []
            for source in sources:
                for name in keys:
                    value = source.get(name)
                    if value not in (None, ""):
                        values.append(value)
            if not keys or not values:
                continue
            clauses = " OR ".join(
                f"n.{quote_ident(name)} IN $values" for name in keys
            )
            cypher = f"MATCH (n:{quoted}) WHERE {clauses} RETURN n.id AS id"
            rows = await self._integration.query(cypher, {"values": values})
            for row in rows:
                found.append({"entity_key": key, "id": extract_node_id(row)})
        return found

    async def _probe_anomalies(
        self, injected: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Probe Neo4j for each injected anomaly kind.

        Args:
            injected: Anomaly dicts with kind and id.

        Returns:
            Subset that the graph still exhibits (surfaced by observation).
        """
        detected: list[dict[str, Any]] = []
        for item in injected:
            kind, ident = _anomaly_key(item)
            if not ident:
                continue
            if kind == "missing":
                rows = await self._integration.query(
                    "MATCH (n {id: $id}) RETURN n.id AS id", {"id": ident}
                )
                if not rows:
                    detected.append(item)
            elif kind == "duplicate":
                rows = await self._integration.query(
                    "MATCH (n {id: $id}) RETURN count(n) AS n", {"id": ident}
                )
                count = int((rows[0] or {}).get("n") or 0) if rows else 0
                if count > 1:
                    detected.append(item)
            elif kind in {"contradictory", "malformed"}:
                # Live probing of semantic contradictions needs domain Cypher;
                # treat a non-empty MATCH on the id as "still present".
                rows = await self._integration.query(
                    "MATCH (n {id: $id}) RETURN n.id AS id", {"id": ident}
                )
                if rows:
                    detected.append(item)
        return detected

    def _require_text(self, value: str, field: str, metric: str) -> None:
        """Raise GraphValidatorError if `value` is empty or whitespace."""
        if not isinstance(value, str) or not value.strip():
            raise GraphValidatorError(
                f"{field} must be a non-empty string",
                evaluator_name=self.name,
                metric=metric,
            )


def _infer_direction(edge: dict[str, Any]) -> str:
    """Infer outgoing/incoming from start/end relative to `from` / `source`."""
    start = str(edge.get("start") or edge.get("from") or edge.get("source") or "")
    end = str(edge.get("end") or edge.get("to") or edge.get("target") or "")
    subject = str(edge.get("node_id") or edge.get("entity_id") or "")
    if subject and start and subject == start:
        return "outgoing"
    if subject and end and subject == end:
        return "incoming"
    if start and end:
        return "outgoing"
    return ""


def _sort_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort events by timestamp when present; otherwise keep given order."""

    def key(item: dict[str, Any]) -> tuple[int, str]:
        stamp = item.get("timestamp") or item.get("ts") or item.get("time")
        if stamp is None:
            return (1, "")
        return (0, str(stamp))

    return sorted(events, key=key)
