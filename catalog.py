import json
import re
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Load catalog
# ---------------------------------------------------------------------------

_CATALOG_PATH = Path(__file__).parent / "catalog_data.json"

def _load_catalog() -> List[Dict[str, Any]]:
    with open(_CATALOG_PATH, "rb") as f:
        raw = f.read()
    items = json.loads(raw, strict=False)
    # Keep only items with a valid link and name
    return [i for i in items if i.get("name") and i.get("link")]

CATALOG: List[Dict[str, Any]] = _load_catalog()

# ---------------------------------------------------------------------------
# Build a compact text representation of each assessment for LLM context
# ---------------------------------------------------------------------------

def _format_item(item: Dict[str, Any]) -> str:
    keys = ", ".join(item.get("keys") or []) or "—"
    duration = item.get("duration") or "—"
    levels = ", ".join(item.get("job_levels") or []) or "—"
    langs = ", ".join((item.get("languages") or [])[:5])
    if len(item.get("languages") or []) > 5:
        langs += f" (+{len(item['languages']) - 5} more)"
    langs = langs or "—"
    desc = (item.get("description") or "")[:200].replace("\n", " ")
    remote = item.get("remote", "")
    adaptive = item.get("adaptive", "")
    flags = []
    if remote == "yes":
        flags.append("remote-enabled")
    if adaptive == "yes":
        flags.append("adaptive")
    flags_str = ", ".join(flags) or "—"

    return (
        f"Name: {item['name']}\n"
        f"URL: {item['link']}\n"
        f"Type: {keys}\n"
        f"Duration: {duration}\n"
        f"Job Levels: {levels}\n"
        f"Languages: {langs}\n"
        f"Flags: {flags_str}\n"
        f"Description: {desc}"
    )

# Pre-build catalog text for injection into prompts
CATALOG_TEXT: str = "\n\n---\n\n".join(_format_item(i) for i in CATALOG)

# ---------------------------------------------------------------------------
# Lookup helpers used for grounding checks
# ---------------------------------------------------------------------------

# Map name (lowercase) -> item
_NAME_INDEX: Dict[str, Dict] = {i["name"].lower(): i for i in CATALOG}
# Map url -> item
_URL_INDEX: Dict[str, Dict] = {i["link"]: i for i in CATALOG}

def get_by_name(name: str) -> Dict | None:
    return _NAME_INDEX.get(name.lower())

def get_by_url(url: str) -> Dict | None:
    return _URL_INDEX.get(url)

def is_valid_url(url: str) -> bool:
    return url in _URL_INDEX

def catalog_summary() -> str:
    """Short summary stats for system prompt."""
    types = {}
    for item in CATALOG:
        for k in (item.get("keys") or []):
            types[k] = types.get(k, 0) + 1
    lines = [f"- {k}: {v} assessments" for k, v in sorted(types.items())]
    return f"Total assessments: {len(CATALOG)}\n" + "\n".join(lines)
