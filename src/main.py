from __future__ import annotations

import json
from pathlib import Path
import urllib.error
import urllib.request

from . import config
from .embeddings import build_embedding_backend
from .graph_index import GraphIndex
from .hnsw_index import HNSWIndex
from .load_data import load_inputs
from .query_parser import build_query_search_text, parse_query
from .ranker import rank_actors
from .transform import transform_actors


def _round_score(value: float) -> float:
    return round(float(value), 4)


def _build_actor_lookup(actors: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(actor["actor_id"]): actor for actor in actors}


def _openrouter_explanation_enabled() -> bool:
    return (
        config.OPENROUTER_EXPLANATIONS_ENABLED
        and bool(config.OPENROUTER_API_KEY)
        and bool(config.OPENROUTER_EXPLANATION_MODEL)
    )


def _extract_json_text(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()

    if stripped.startswith("{") or stripped.startswith("["):
        return stripped

    start_candidates = [index for index in (stripped.find("{"), stripped.find("[")) if index != -1]
    if not start_candidates:
        return stripped
    start = min(start_candidates)

    end_brace = stripped.rfind("}")
    end_bracket = stripped.rfind("]")
    end = max(end_brace, end_bracket)
    if end >= start:
        return stripped[start : end + 1]
    return stripped


def _normalize_explanation_response(
    structured: object,
    result_items: list[dict[str, object]],
) -> dict[str, str]:
    actor_ids = [str(item["actor_id"]) for item in result_items]

    if isinstance(structured, dict):
        explanations = structured.get("explanations")
        if isinstance(explanations, dict):
            return {
                str(actor_id).strip(): " ".join(str(explanation).split())
                for actor_id, explanation in explanations.items()
                if str(actor_id).strip() and " ".join(str(explanation).split())
            }
        if isinstance(explanations, list):
            explanation_map: dict[str, str] = {}
            for index, item in enumerate(explanations):
                if isinstance(item, dict):
                    actor_id = str(item.get("actor_id", "")).strip()
                    explanation = " ".join(str(item.get("explanation", "")).split())
                    if actor_id and explanation:
                        explanation_map[actor_id] = explanation
                        continue
                if isinstance(item, str) and index < len(actor_ids):
                    explanation = " ".join(item.split())
                    if explanation:
                        explanation_map[actor_ids[index]] = explanation
            return explanation_map

    if isinstance(structured, list):
        explanation_map: dict[str, str] = {}
        for index, item in enumerate(structured):
            if index >= len(actor_ids):
                break
            if isinstance(item, dict):
                explanation = " ".join(str(item.get("explanation", "")).split())
            else:
                explanation = " ".join(str(item).split())
            if explanation:
                explanation_map[actor_ids[index]] = explanation
        return explanation_map

    if isinstance(structured, str):
        explanation = " ".join(structured.split())
        if explanation and actor_ids:
            return {actor_ids[0]: explanation}

    return {}


def _build_openrouter_explanation_payload(
    query: str,
    result_items: list[dict[str, object]],
    actor_lookup: dict[str, dict[str, object]],
) -> bytes:
    compact_results: list[dict[str, object]] = []
    for result in result_items:
        actor = actor_lookup.get(str(result["actor_id"]), {})
        compact_results.append(
            {
                "actor_id": result["actor_id"],
                "name": result["name"],
                "headline": actor.get("headline", ""),
                "current_title": actor.get("current_title", ""),
                "current_company": actor.get("current_company", ""),
                "score": result["score"],
                "vector_score": result["vector_score"],
                "graph_score": result["graph_score"],
                "structured_boost": result["structured_boost"],
                "matched_graph_edges": result["matched_graph_edges"],
                "matched_tags": result["matched_tags"],
            }
        )

    prompt_data = {"query": query, "results": compact_results}
    return json.dumps(
        {
            "model": config.OPENROUTER_EXPLANATION_MODEL,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You write concise retrieval debug explanations. "
                        "Use only the provided metadata. "
                        "Do not invent biography details or ranking logic beyond the fields given. "
                        "Return strict JSON with shape "
                        '{"explanations":[{"actor_id":"...","explanation":"..."}]}. '
                        "Each explanation must be one sentence and should mention the strongest graph, tag, "
                        "or semantic match when present."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt_data, ensure_ascii=True)},
            ],
        }
    ).encode("utf-8")


def _maybe_generate_openrouter_explanations(
    query: str,
    result_items: list[dict[str, object]],
    actor_lookup: dict[str, dict[str, object]],
) -> dict[str, str]:
    if not _openrouter_explanation_enabled() or not result_items:
        return {}

    request = urllib.request.Request(
        config.OPENROUTER_CHAT_BASE_URL,
        data=_build_openrouter_explanation_payload(query, result_items, actor_lookup),
        headers={
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.OPENROUTER_TIMEOUT_SECONDS) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {}

    try:
        content = parsed["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        structured = json.loads(_extract_json_text(content))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return {}
    return _normalize_explanation_response(structured, result_items)


def run() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_actors, queries = load_inputs()
    actors = transform_actors(raw_actors)
    actor_lookup = _build_actor_lookup(actors)

    documents = [actor["search_text"] for actor in actors]
    embedding_backend = build_embedding_backend()
    hnsw_index = HNSWIndex(embedding_backend)
    hnsw_index.fit(documents)
    graph_index = GraphIndex()
    graph_index.fit(actors)

    results_payload: list[dict[str, object]] = []
    debug_payload: list[dict[str, object]] = []

    for query in queries:
        parsed_query = parse_query(query)
        query_text = build_query_search_text(parsed_query)
        vector_scores = hnsw_index.score(query_text)
        graph_matches = graph_index.score(parsed_query)
        ranked_results = rank_actors(parsed_query, actors, vector_scores, graph_matches)

        results_payload.append(
            {
                "query": query,
                "results": [
                    {"actor_id": item["actor_id"], "score": _round_score(item["score"])}
                    for item in ranked_results
                ],
            }
        )

        openrouter_explanations = _maybe_generate_openrouter_explanations(
            query,
            ranked_results,
            actor_lookup,
        )

        debug_payload.append(
            {
                "query": query,
                "results": [
                    {
                        "actor_id": item["actor_id"],
                        "name": item["name"],
                        "score": _round_score(item["score"]),
                        "vector_score": _round_score(item["vector_score"]),
                        "graph_score": _round_score(item["graph_score"]),
                        "structured_boost": _round_score(item["structured_boost"]),
                        "matched_graph_edges": item["matched_graph_edges"],
                        "matched_tags": item["matched_tags"],
                        "explanation": openrouter_explanations.get(item["actor_id"], item["explanation"]),
                    }
                    for item in ranked_results
                ],
            }
        )

    return results_payload, debug_payload


def write_outputs(results_payload: list[dict[str, object]], debug_payload: list[dict[str, object]]) -> None:
    Path(config.OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.RESULTS_PATH).write_text(
        json.dumps(results_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    Path(config.DEBUG_RESULTS_PATH).write_text(
        json.dumps(debug_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def main() -> None:
    results_payload, debug_payload = run()
    write_outputs(results_payload, debug_payload)
    print(f"Wrote {config.RESULTS_PATH}")
    print(f"Wrote {config.DEBUG_RESULTS_PATH}")


if __name__ == "__main__":
    main()
