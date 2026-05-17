from __future__ import annotations

from typing import Any

from . import config


def _normalize_company(value: str) -> str:
    return " ".join(value.lower().replace("&", "and").split())


def _normalize_location(value: str) -> str:
    lowered = value.lower()
    for source, target in sorted(
        config.LOCATION_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        lowered = lowered.replace(source, target)
    return " ".join(lowered.split())


def compute_structured_boost(parsed_query: dict[str, Any], actor: dict[str, Any]) -> tuple[float, list[str]]:
    tags = set(actor.get("tags") or [])
    companies = {_normalize_company(company) for company in actor.get("companies") or []}
    titles = " ".join(actor.get("titles") or []).lower()
    location = _normalize_location(actor.get("normalized_location") or actor.get("location") or "")
    headline = (actor.get("headline") or "").lower()
    raw_text = (actor.get("raw_text") or "").lower()

    boost = 0.0
    matched: list[str] = []

    if "founder" in parsed_query.get("roles", []) and tags.intersection({"founder", "cofounder"}):
        boost += 0.15
        matched.append("founder")

    if parsed_query.get("locations"):
        if any(location_name in location or location_name in tags for location_name in parsed_query["locations"]):
            boost += 0.15
            matched.extend(parsed_query["locations"])

    query_companies = parsed_query.get("companies", [])
    if query_companies and any(company in companies or company in raw_text for company in query_companies):
        boost += 0.20
        matched.extend(query_companies)

    if "marketing" in parsed_query.get("roles", []) and (
        "marketing" in titles or "marketing" in headline
    ):
        boost += 0.10
        matched.append("marketing")

    if "fintech" in parsed_query.get("domains", []) and "fintech" in tags:
        boost += 0.10
        matched.append("fintech")

    if "machine_learning" in parsed_query.get("signals", []) and tags.intersection(
        {"machine_learning", "ml"}
    ):
        boost += 0.10
        matched.append("machine_learning")

    if set(parsed_query.get("signals", [])).intersection({"mentor", "startup"}) and tags.intersection(
        {"mentor", "advisor", "founder", "cofounder", "startup"}
    ):
        boost += 0.15
        matched.append("mentor_startup")

    if set(parsed_query.get("signals", [])).intersection({"computer_vision", "vision_ai"}) and tags.intersection(
        {"computer_vision", "vision_ai", "perception"}
    ):
        boost += 0.15
        matched.append("computer_vision")

    if "healthcare" in parsed_query.get("domains", []) and "healthcare" in tags:
        boost += 0.10
        matched.append("healthcare")

    return min(1.0, boost), sorted(set(matched))


def rank_actors(
    parsed_query: dict[str, Any],
    actors: list[dict[str, Any]],
    vector_scores: list[float],
    graph_matches: dict[str, Any],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    limit = top_k or config.TOP_K
    ranked: list[dict[str, Any]] = []
    for actor, vector_score in zip(actors, vector_scores):
        graph_match = graph_matches.get(actor["actor_id"])
        graph_score = graph_match.graph_score if graph_match else 0.0
        matched_graph_edges = graph_match.matched_graph_edges if graph_match else []
        structured_boost, matched_tags = compute_structured_boost(parsed_query, actor)
        final_score = (
            config.VECTOR_WEIGHT * vector_score
            + config.GRAPH_WEIGHT * graph_score
            + config.STRUCTURED_WEIGHT * structured_boost
        )
        explanation = "Matched " + ", ".join(matched_graph_edges) if matched_graph_edges else "Matched semantic intent."
        ranked.append(
            {
                "actor_id": actor["actor_id"],
                "name": actor["name"],
                "score": min(1.0, max(0.0, final_score)),
                "vector_score": min(1.0, max(0.0, vector_score)),
                "graph_score": min(1.0, max(0.0, graph_score)),
                "structured_boost": structured_boost,
                "matched_graph_edges": matched_graph_edges,
                "matched_tags": matched_tags,
                "explanation": explanation,
            }
        )

    ranked.sort(key=lambda item: (-item["score"], -item["graph_score"], item["actor_id"]))
    return ranked[:limit]
