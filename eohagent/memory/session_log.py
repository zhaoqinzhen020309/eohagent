import json
import time
from pathlib import Path


class SessionLog:
    def __init__(self, output_dir: Path):
        self.path = output_dir / "session_log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict) -> None:
        record = {"t": time.time(), **event}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def recent(self, limit: int = 50) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]
