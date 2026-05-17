from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from . import config


RELATION_NODE_LABELS = {
    "WORKED_AT": "Company",
    "LOCATED_IN": "Location",
    "HAS_ROLE": "Role",
    "HAS_DOMAIN": "Domain",
    "HAS_SIGNAL": "Signal",
    "STUDIED_AT": "School",
}

DOMAIN_TAGS = {"fintech", "healthcare", "machine_learning", "computer_vision", "vision_ai", "ai"}


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned


def _normalize_company(value: str) -> str:
    return _normalize_text(value.replace("&", "and"))


def _normalize_school(value: str) -> str:
    return _normalize_text(value)


def _derive_roles(actor: dict[str, Any]) -> set[str]:
    titles = " ".join(actor.get("titles") or []).lower()
    headline = (actor.get("headline") or "").lower()
    combined = f"{titles} {headline}"
    roles: set[str] = set()
    if any(term in combined for term in ("founder", "co-founder", "cofounder", "ceo")):
        roles.add("founder")
    if "marketing" in combined:
        roles.add("marketing")
    if "growth" in combined:
        roles.add("growth")
    return roles


def _derive_domains(actor: dict[str, Any]) -> set[str]:
    tags = set(actor.get("tags") or [])
    domains = tags.intersection(DOMAIN_TAGS)
    if "stripe" in tags or "fintech" in tags:
        domains.add("fintech")
    return domains


def _build_actor_edges(actor: dict[str, Any]) -> dict[str, set[str]]:
    return {
        "WORKED_AT": {_normalize_company(company) for company in actor.get("companies") or [] if company},
        "LOCATED_IN": {actor.get("normalized_location")} if actor.get("normalized_location") else set(),
        "HAS_ROLE": _derive_roles(actor),
        "HAS_DOMAIN": _derive_domains(actor),
        "HAS_SIGNAL": {_normalize_text(tag) for tag in actor.get("tags") or [] if tag},
        "STUDIED_AT": {_normalize_school(school) for school in actor.get("schools") or [] if school},
    }


def _build_query_constraints(parsed_query: dict[str, Any]) -> list[tuple[str, str]]:
    constraints: list[tuple[str, str]] = []
    constraints.extend(("WORKED_AT", _normalize_company(company)) for company in parsed_query.get("companies", []))
    constraints.extend(("LOCATED_IN", _normalize_text(location)) for location in parsed_query.get("locations", []))
    constraints.extend(("HAS_DOMAIN", _normalize_text(domain)) for domain in parsed_query.get("domains", []))
    for role in parsed_query.get("roles", []):
        if role == "founder":
            constraints.append(("HAS_SIGNAL", "founder"))
        elif role in {"mentor", "advisor"}:
            constraints.append(("HAS_SIGNAL", _normalize_text(role)))
        else:
            constraints.append(("HAS_ROLE", _normalize_text(role)))
    constraints.extend(("HAS_SIGNAL", _normalize_text(signal)) for signal in parsed_query.get("signals", []))

    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for item in constraints:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


@dataclass
class GraphMatch:
    graph_score: float
    matched_graph_edges: list[str]


class GraphIndex:
    def __init__(self) -> None:
        self.backend = "memgraph"
        self.actor_edges: dict[str, dict[str, set[str]]] = {}
        self.actor_ids: list[str] = []
        self.driver = None
        if not config.MEMGRAPH_USE:
            raise RuntimeError("Memgraph is required. Set MEMGRAPH_USE=true and start the Memgraph container.")

        auth = None
        if config.MEMGRAPH_USERNAME:
            auth = (config.MEMGRAPH_USERNAME, config.MEMGRAPH_PASSWORD)
        self.driver = GraphDatabase.driver(
            f"bolt://{config.MEMGRAPH_HOST}:{config.MEMGRAPH_PORT}",
            auth=auth,
        )
        self.driver.verify_connectivity()

    def fit(self, actors: list[dict[str, Any]]) -> None:
        self.actor_ids = [actor["actor_id"] for actor in actors]
        self.actor_edges = {actor["actor_id"]: _build_actor_edges(actor) for actor in actors}
        self._populate_memgraph(actors)

    def _populate_memgraph(self, actors: list[dict[str, Any]]) -> None:
        assert self.driver is not None
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            for actor in actors:
                actor_id = actor["actor_id"]
                session.run(
                    "MERGE (p:Person {actor_id: $actor_id}) SET p.name = $name",
                    actor_id=actor_id,
                    name=actor["name"],
                )
                for relation, values in self.actor_edges[actor_id].items():
                    label = RELATION_NODE_LABELS[relation]
                    for value in values:
                        session.run(
                            f"""
                            MATCH (p:Person {{actor_id: $actor_id}})
                            MERGE (n:{label} {{name: $value}})
                            MERGE (p)-[:{relation}]->(n)
                            """,
                            actor_id=actor_id,
                            value=value,
                        )

    def score(self, parsed_query: dict[str, Any]) -> dict[str, GraphMatch]:
        constraints = _build_query_constraints(parsed_query)
        return self._score_with_memgraph(constraints)

    def _score_with_memgraph(self, constraints: list[tuple[str, str]]) -> dict[str, GraphMatch]:
        assert self.driver is not None
        total_constraints = len(constraints)
        matches: dict[str, list[str]] = {actor_id: [] for actor_id in self.actor_ids}
        try:
            with self.driver.session() as session:
                for relation, value in constraints:
                    label = RELATION_NODE_LABELS[relation]
                    result = session.run(
                        f"""
                        MATCH (p:Person)-[:{relation}]->(n:{label} {{name: $value}})
                        RETURN p.actor_id AS actor_id
                        """,
                        value=value,
                    )
                    for row in result:
                        actor_id = row["actor_id"]
                        if actor_id in matches:
                            matches[actor_id].append(f"{relation}: {value}")
        except (Neo4jError, ServiceUnavailable):
            raise RuntimeError("Memgraph query failed during graph scoring.") from None

        return {
            actor_id: GraphMatch(
                graph_score=(len(edges) / total_constraints) if total_constraints else 0.0,
                matched_graph_edges=edges,
            )
            for actor_id, edges in matches.items()
        }
