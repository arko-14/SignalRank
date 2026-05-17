from __future__ import annotations

import re
from typing import Any

from . import config


COMPANY_KEYWORDS = {
    "google": "google",
    "stripe": "stripe",
    "plaid": "plaid",
    "robinhood": "robinhood",
    "square": "square",
}

DOMAIN_KEYWORDS = {
    "fintech": "fintech",
    "healthcare": "healthcare",
    "health tech": "healthcare",
    "healthtech": "healthcare",
    "ai": "ai",
    "machine_learning": "machine_learning",
    "computer_vision": "computer_vision",
    "vision_ai": "vision_ai",
}

ROLE_KEYWORDS = {
    "founder": "founder",
    "co-founder": "founder",
    "cofounder": "founder",
    "marketing": "marketing",
    "growth": "growth",
    "mentor": "mentor",
    "advisor": "advisor",
}

SIGNAL_KEYWORDS = {
    "startup": "startup",
    "machine_learning": "machine_learning",
    "computer_vision": "computer_vision",
    "vision_ai": "vision_ai",
    "mentor": "mentor",
    "advisor": "advisor",
}


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_query_text(query: str) -> str:
    normalized = query.strip().lower()
    for source, target in sorted(
        config.QUERY_NORMALIZATIONS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized)
    normalized = re.sub(r"[^a-z0-9_+\-/ ]+", " ", normalized)
    return _normalize_whitespace(normalized)


def _collect_locations(normalized_query: str) -> list[str]:
    locations: list[str] = []
    for source, canonical in config.LOCATION_ALIASES.items():
        if canonical in normalized_query or re.search(rf"\b{re.escape(source)}\b", normalized_query):
            locations.append(canonical)
    return sorted(set(locations))


def _collect_companies(normalized_query: str) -> list[str]:
    companies = {canonical for keyword, canonical in COMPANY_KEYWORDS.items() if keyword in normalized_query}
    match = re.search(r"worked at ([a-z0-9&._ -]+)", normalized_query)
    if match:
        phrase = match.group(1).strip()
        first_word = phrase.split()[0] if phrase else ""
        if first_word:
            companies.add(first_word)
    return sorted(companies)


def _collect_keywords(normalized_query: str, mapping: dict[str, str]) -> list[str]:
    found = {canonical for keyword, canonical in mapping.items() if keyword in normalized_query}
    return sorted(found)


def parse_query(query: str) -> dict[str, Any]:
    normalized_query = normalize_query_text(query)
    locations = _collect_locations(normalized_query)
    companies = _collect_companies(normalized_query)
    domains = _collect_keywords(normalized_query, DOMAIN_KEYWORDS)
    roles = _collect_keywords(normalized_query, ROLE_KEYWORDS)
    signals = _collect_keywords(normalized_query, SIGNAL_KEYWORDS)

    if "marketing" in roles and "growth" not in roles and "growth" in normalized_query:
        roles.append("growth")
    if "machine_learning" in domains and "machine_learning" not in signals:
        signals.append("machine_learning")
    if "computer_vision" in domains and "computer_vision" not in signals:
        signals.append("computer_vision")
    if "vision_ai" in domains and "vision_ai" not in signals:
        signals.append("vision_ai")

    return {
        "raw_query": query,
        "normalized_query": normalized_query,
        "locations": sorted(set(locations)),
        "companies": sorted(set(companies)),
        "domains": sorted(set(domains)),
        "roles": sorted(set(roles)),
        "signals": sorted(set(signals)),
    }


def build_query_search_text(parsed_query: dict[str, Any]) -> str:
    parts = [
        f"Query: {parsed_query['raw_query']}",
        f"Normalized Query: {parsed_query['normalized_query']}",
        f"Locations: {', '.join(parsed_query['locations']) or 'none'}",
        f"Companies: {', '.join(parsed_query['companies']) or 'none'}",
        f"Domains: {', '.join(parsed_query['domains']) or 'none'}",
        f"Roles: {', '.join(parsed_query['roles']) or 'none'}",
        f"Signals: {', '.join(parsed_query['signals']) or 'none'}",
    ]
    return "\n".join(parts)
