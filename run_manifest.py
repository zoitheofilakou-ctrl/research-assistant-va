import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from project_paths import BASE_DIR, RUN_MANIFEST_DIR, ensure_dir


class RunManifest:
    def __init__(self, script_name: str):
        self.script_name = script_name
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.events = []
        self.summary: Dict[str, Any] = {}

    def add_event(self, action: str, path: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        event = {
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if path:
            event["path"] = os.path.relpath(path, BASE_DIR)
        if details:
            event["details"] = details
        self.events.append(event)

    def set_summary(self, **kwargs):
        self.summary.update(kwargs)

    def write(self) -> str:
        ensure_dir(RUN_MANIFEST_DIR)
        manifest_path = os.path.join(RUN_MANIFEST_DIR, f"{self.script_name}.json")
        payload = {
            "script_name": self.script_name,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "summary": self.summary,
            "events": self.events,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return manifest_path
