from __future__ import annotations

from typing import Any

from . import config
from .load_data import resolve_actor_id


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    return " ".join(str(value).split())


def _normalize_company(value: str) -> str:
    return _clean_string(value).lower().replace("&", "and")


def _normalize_location(location: str) -> str:
    normalized = _clean_string(location).lower()
    if "bengaluru" in normalized or "bangalore" in normalized or "blr" in normalized:
        return "bangalore"
    if "san francisco" in normalized or normalized == "sf":
        return "san_francisco"
    return "_".join(normalized.split())


def _collect_accomplishments(professional: dict[str, Any]) -> list[str]:
    accomplishments = professional.get("accomplishments") or {}
    collected: list[str] = []
    for category in ("certifications", "courses", "honors", "languages", "projects", "publications"):
        for item in accomplishments.get(category) or []:
            if isinstance(item, dict):
                name = _clean_string(item.get("name") or item.get("title"))
                authority = _clean_string(
                    item.get("authority") or item.get("issuer") or item.get("publisher")
                )
                detail = f"{name} ({authority})".strip()
                collected.append(detail if authority else name)
            else:
                collected.append(_clean_string(item))
    return [item for item in collected if item]


def _build_experience_lines(work_experience: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for role in work_experience:
        title = _clean_string(role.get("title"))
        company = _clean_string(role.get("company_name") or role.get("company"))
        description = _clean_string(role.get("description"))
        fragments = [fragment for fragment in (title, f"at {company}" if company else "", description) if fragment]
        if fragments:
            lines.append(" ".join(fragments))
    return lines


def _build_education_lines(education_entries: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for entry in education_entries:
        school = _clean_string(entry.get("school"))
        degree = _clean_string(entry.get("degree"))
        field = _clean_string(entry.get("field_of_study"))
        description = _clean_string(entry.get("description"))
        fragments = [school, degree, field, description]
        line = ", ".join(fragment for fragment in fragments if fragment)
        if line:
            lines.append(line)
    return lines


def _build_schools(education_entries: list[dict[str, Any]]) -> list[str]:
    schools = [
        school
        for school in dict.fromkeys(
            _clean_string(entry.get("school")) for entry in education_entries if _clean_string(entry.get("school"))
        )
    ]
    return schools


def generate_tags(actor: dict[str, Any], location: str, companies: list[str], titles: list[str]) -> list[str]:
    profile = actor.get("profile") or {}
    professional = actor.get("professional") or {}
    experience_lines = _build_experience_lines(professional.get("work_experience") or [])
    education_lines = _build_education_lines(professional.get("education") or [])
    accomplishment_lines = _collect_accomplishments(professional)
    text = " ".join(
        [
            _clean_string(profile.get("headline")),
            _clean_string(profile.get("bio")),
            location,
            " ".join(companies),
            " ".join(titles),
            " ".join(experience_lines),
            " ".join(education_lines),
            " ".join(accomplishment_lines),
        ]
    ).lower()

    rules = {
        "founder": ("founder", "co-founder", "cofounder", "founded", "ceo"),
        "cofounder": ("co-founder", "cofounder"),
        "startup": ("startup", "start-up", "founder", "seed", "series a"),
        "fundraising": ("raised", "funding", "seed", "series a", "series b", "investor", "venture"),
        "yc": ("yc", "y combinator"),
        "sequoia": ("sequoia",),
        "fintech": ("fintech", "payments", "banking", "wealthtech", "lending"),
        "marketing": ("marketing", "brand", "demand gen", "go to market", "gtm"),
        "growth": ("growth", "acquisition", "demand gen", "performance marketing"),
        "machine_learning": ("machine learning", "machine_learning", "ml", "deep learning"),
        "ml": (" machine learning ", " ml ", "mlops", "ai/ml"),
        "computer_vision": ("computer vision", "computer_vision", "vision", "perception", "face id", "autopilot"),
        "vision_ai": ("vision ai", "vision_ai", "computer vision", "perception"),
        "mentor": ("mentor", "mentoring", "office hours", "helping founders"),
        "advisor": ("advisor", "advising", "advisory"),
        "healthcare": ("healthcare", "health tech", "healthtech", "medtech", "clinical"),
        "google": ("google",),
        "stripe": ("stripe",),
    }

    tags: set[str] = set()
    padded_text = f" {text} "
    for tag, keywords in rules.items():
        if any(keyword in padded_text for keyword in keywords):
            tags.add(tag)

    normalized_location = _normalize_location(location)
    if "bangalore" in normalized_location:
        tags.add("bangalore")
    if "san_francisco" in normalized_location:
        tags.add("san_francisco")

    normalized_companies = {_normalize_company(company) for company in companies}
    if "google" in normalized_companies:
        tags.add("google")
    if "stripe" in normalized_companies:
        tags.add("stripe")

    return sorted(tags)


def transform_actor(actor: dict[str, Any]) -> dict[str, Any]:
    profile = actor.get("profile") or {}
    professional = actor.get("professional") or {}
    current_position = professional.get("current_position") or {}
    work_experience = professional.get("work_experience") or []
    education_entries = professional.get("education") or []

    actor_id = _clean_string(actor.get("resolved_actor_id")) or resolve_actor_id(actor)
    name = _clean_string(profile.get("name"))
    location = _clean_string(profile.get("location"))
    normalized_location = _normalize_location(location)
    headline = _clean_string(profile.get("headline"))
    bio = _clean_string(profile.get("bio"))
    current_title = _clean_string(current_position.get("title"))
    current_company = _clean_string(current_position.get("company"))

    companies = [
        company
        for company in dict.fromkeys(
            _clean_string(item.get("company_name") or item.get("company"))
            for item in work_experience
            if _clean_string(item.get("company_name") or item.get("company"))
        )
    ]
    if current_company and current_company not in companies:
        companies.insert(0, current_company)

    titles = [
        title
        for title in dict.fromkeys(
            _clean_string(item.get("title")) for item in work_experience if _clean_string(item.get("title"))
        )
    ]
    if current_title and current_title not in titles:
        titles.insert(0, current_title)

    education = _build_education_lines(education_entries)
    schools = _build_schools(education_entries)
    experience_lines = _build_experience_lines(work_experience)
    accomplishment_lines = _collect_accomplishments(professional)
    tags = generate_tags(actor, location, companies, titles)

    raw_text_parts = [
        name,
        location,
        headline,
        bio,
        current_title,
        current_company,
        " ".join(companies),
        " ".join(titles),
        " ".join(experience_lines),
        " ".join(education),
        " ".join(accomplishment_lines),
        " ".join(tags),
    ]
    raw_text = " ".join(part for part in raw_text_parts if part)

    search_lines = [
        f"Name: {name}",
        f"Location: {location}",
        f"Headline: {headline}",
        f"Current Role: {current_title}{' at ' + current_company if current_company else ''}",
        f"Bio: {bio}",
        "Experience:",
    ]
    search_lines.extend(f"- {line}" for line in experience_lines or ["No listed experience"])
    search_lines.append("Education:")
    search_lines.extend(f"- {line}" for line in education or ["No listed education"])
    if accomplishment_lines:
        search_lines.append("Accomplishments:")
        search_lines.extend(f"- {line}" for line in accomplishment_lines)
    search_lines.append(f"Signals: {', '.join(tags)}")

    return {
        "actor_id": actor_id,
        "name": name,
        "location": location,
        "normalized_location": normalized_location,
        "headline": headline,
        "bio": bio,
        "current_title": current_title,
        "current_company": current_company,
        "companies": companies,
        "titles": titles,
        "education": education,
        "schools": schools,
        "raw_text": raw_text,
        "search_text": "\n".join(search_lines),
        "tags": tags,
    }


def transform_actors(actors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [transform_actor(actor) for actor in actors]
