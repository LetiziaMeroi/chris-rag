import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_DIR = Path(
    os.getenv(
        "CHRIS_LOG_DIR",
        "logs",
    )
)

LOG_PATH = (
    LOG_DIR
    / "queries.jsonl"
)


_lock = Lock()


def log_query(
    *,
    query: str,
    route: str,
    status: str,
    answer: str | None,
    latency_seconds: float,
):
    """
    Append one structured query log entry.

    Evidence and retrieved document contents are intentionally
    not stored here to keep logs compact and reduce unnecessary
    duplication of potentially sensitive data.
    """

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "query": query,

        "route": route,

        "status": status,

        "latency_seconds": round(
            latency_seconds,
            3,
        ),

        "answer": answer,
    }

    with _lock:
        with LOG_PATH.open(
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
