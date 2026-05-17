from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from . import config


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    return " ".join(str(value).split())


def _extract_oid(value: Any) -> str:
    if isinstance(value, dict):
        oid = value.get("$oid")
        if oid:
            return _clean_string(oid)
    if isinstance(value, str):
        return _clean_string(value)
    return ""


def _primary_platform_id(actor: dict[str, Any]) -> str:
    identities = actor.get("platform_identities") or []
    primary = next((item for item in identities if item.get("is_primary")), None)
    chosen = primary or (identities[0] if identities else {})
    return _clean_string(chosen.get("platform_id"))


def resolve_actor_id(actor: dict[str, Any]) -> str:
    professional = actor.get("professional") or {}
    current_position = professional.get("current_position") or {}
    current_oid = _extract_oid(current_position.get("actor_id"))
    if current_oid:
        return current_oid

    work_experience = professional.get("work_experience") or []
    if work_experience:
        experience_oid = _extract_oid((work_experience[0] or {}).get("actor_id"))
        if experience_oid:
            return experience_oid

    platform_id = _primary_platform_id(actor)
    if platform_id:
        return platform_id

    name = _clean_string((actor.get("profile") or {}).get("name"))
    fallback_seed = f"{name}|{platform_id}"
    return hashlib.sha1(fallback_seed.encode("utf-8")).hexdigest()[:16]


def load_actor_profiles(path: Path | None = None) -> list[dict[str, Any]]:
    source = path or config.ACTORS_PATH
    raw = json.loads(Path(source).read_text(encoding="utf-8"))
    profiles: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    for actor in raw:
        copied = dict(actor)
        resolved_actor_id = resolve_actor_id(actor)
        collision_count = seen_ids.get(resolved_actor_id, 0)
        if collision_count:
            name = _clean_string((actor.get("profile") or {}).get("name"))
            platform_id = _primary_platform_id(actor)
            collision_seed = f"{resolved_actor_id}|{name}|{platform_id}|{collision_count}"
            suffix = hashlib.sha1(collision_seed.encode("utf-8")).hexdigest()[:8]
            resolved_actor_id = f"{resolved_actor_id}-{suffix}"
        seen_ids[resolve_actor_id(actor)] = collision_count + 1
        copied["resolved_actor_id"] = resolved_actor_id
        profiles.append(copied)
    return profiles


def _choose_query_column(headers: list[str]) -> str | None:
    lowered = {header.strip().lower(): header for header in headers}
    for candidate in ("query", "queries", "text"):
        if candidate in lowered:
            return lowered[candidate]
    return headers[0] if headers else None


def load_queries(path: Path | None = None) -> list[str]:
    source = path or config.QUERIES_PATH
    with Path(source).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        query_column = _choose_query_column(reader.fieldnames or [])
        if not query_column:
            return []
        queries: list[str] = []
        for row in reader:
            query = _clean_string(row.get(query_column))
            if query:
                queries.append(query)
    return queries


def load_inputs() -> tuple[list[dict[str, Any]], list[str]]:
    return load_actor_profiles(), load_queries()
