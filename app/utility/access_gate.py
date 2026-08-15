"""Simple demo entry gate — requires a keyword plus captures visitor
name/company. Not real auth, just a lightweight filter + visitor log."""

import time
import json
import os
from pathlib import Path
from fastapi import Request

VISITOR_LOG_PATH = Path(__file__).parent.parent / "data" / "demo_visitors.jsonl"

# Set in Secret Manager / .env — pick something you'll share on your resume,
# LinkedIn, or application (e.g. "SIMPSONS2026" or "HIREME")
DEMO_KEYWORD = os.environ.get("DEMO_KEYWORD", "SIMPSONS")


def check_keyword(entered: str) -> bool:
    return entered.strip().upper() == DEMO_KEYWORD.upper()


def log_visitor(name: str, company: str, request: Request) -> None:
    """Append visitor info to a simple local log. Best-effort — never blocks access."""
    VISITOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "name": name.strip()[:100],
        "company": company.strip()[:100],
        "timestamp": int(time.time()),
        "ip": request.client.host if request.client else "unknown",
    }
    try:
        with open(VISITOR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
