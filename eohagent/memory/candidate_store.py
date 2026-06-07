import json
from pathlib import Path

from eohagent.core.candidate import Candidate


class CandidateStore:
    def __init__(self, output_dir: Path):
        self.path = output_dir / "candidate_store.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def all(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, candidate: Candidate) -> None:
        data = self.all()
        data[candidate.id] = candidate.to_dict()
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
