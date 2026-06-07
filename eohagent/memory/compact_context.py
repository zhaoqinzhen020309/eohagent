import json
from pathlib import Path


class CompactContext:
    def __init__(self, output_dir: Path):
        self.path = output_dir / "compact_context.json"

    def write_default(self, problem_card: dict, seed) -> None:
        self.write(
            {
                "problem_card": problem_card,
                "current_best": {
                    "id": seed.id,
                    "mechanism": seed.mechanism_summary,
                    "objective": seed.objective,
                },
                "recent_lessons": [],
                "active_design_rules": [
                    "Preserve baseline core unless evidence suggests otherwise.",
                    "Use controlled variation, not full rewrite.",
                ],
                "next_recommended_focus": ["Try a small controlled variation of the baseline seed."],
            }
        )

    def read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
