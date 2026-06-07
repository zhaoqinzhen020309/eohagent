from pathlib import Path


class SkillRegistry:
    def __init__(self, root: Path):
        self.root = root
        self.skills_dir = root / "eohagent" / "skills"
        self._metadata = [
            {"name": "mechanism_decomposition", "description": "Analyze a parent heuristic and produce a mechanism-level experiment plan before modifying it."},
            {"name": "controlled_variation", "description": "Implement one mechanism-level experiment according to a design_mode and evidence-backed intent."},
            {"name": "contract_check", "description": "Check syntax, signature, runtime, output shape, and compatibility."},
            {"name": "context_compaction", "description": "Compact state and recent events into future context."},
            {"name": "notebook_update", "description": "Convert raw outcomes into bounded mechanism-level design memory."},
        ]

    def metadata(self) -> list[dict]:
        return self._metadata

    def load(self, name: str) -> str:
        path = self.skills_dir / name / "SKILL.md"
        if not path.exists():
            raise ValueError(f"Unknown skill: {name}")
        return path.read_text(encoding="utf-8")
