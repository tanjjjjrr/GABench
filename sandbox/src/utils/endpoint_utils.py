from __future__ import annotations


def normalize_xxx_base_url(url: str | None) -> str:
    """Normalize xxx base URL to the /api/v1 form.

    Examples:
      xxx    -> xxx
      xxx   -> xxx
      xxx -> xxx
    """
    if not url:
        return "xxx"

    normalized = url.strip().rstrip("/")
    if normalized.endswith("/api"):
        return normalized + "/v1"
    return normalized


def normalize_xxx_base_url_for_xxx(url: str | None) -> str:
    return normalize_xxx_base_url(url)


def normalize_xxx_base_url_for_xxx(url: str | None) -> str:
    if not url:
        return "xxx"

    normalized = url.strip().rstrip("/")
    if normalized.endswith("/api/v1"):
        return normalized[: -len("/v1")]
    return normalized
