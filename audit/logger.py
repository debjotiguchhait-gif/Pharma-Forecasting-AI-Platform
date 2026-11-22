
import json
import os
from pathlib import Path
from datetime import datetime, timezone


def record_event(user_id, action, status, details=None):
    audit_file = Path(
        os.environ.get(
            "PHARMA_AUDIT_PATH",
            "outputs/audit_log.jsonl",
        )
    )

    audit_file.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "action": action,
        "status": status,
        "details": details or {},
    }

    with audit_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")
