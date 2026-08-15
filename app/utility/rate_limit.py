"""Rate limiting for chat endpoints — protects against rapid-fire/bot traffic."""

import time
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import HTTPException

# IP-based limiter (per-minute cap, backed by slowapi)
limiter = Limiter(key_func=get_remote_address)

# Session-based minimum-interval guard (in-memory; fine for single Cloud Run instance)
_last_request_times: dict[str, float] = {}
MIN_SECONDS_BETWEEN_REQUESTS = 2.0


def enforce_session_interval(session_id: str) -> None:
    """Raise 429 if this session sent a message too recently."""
    now = time.time()
    last_time = _last_request_times.get(session_id)

    if last_time is not None and (now - last_time) < MIN_SECONDS_BETWEEN_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Too many requests — please wait a moment before sending another message.",
        )

    _last_request_times[session_id] = now
