import json

from eohagent.core.candidate import Candidate
from eohagent.core.config import resolve_path
from eohagent.core.mechanism import MechanismHypothesis, canonicalize_family, infer_mechanism_components
from eohagent.problems.tsp_gls_arch_spec.spec_schema import (
    build_arch_spec_code,
    canonical_arch_spec,
    describe_arch_spec,
    mechanism_family_from_spec,
)
from eohagent.problems.tsp_gls_solver_tree.tree_schema import (
    build_solver_tree_code,
    canonical_solver_tree,
    describe_solver_tree,
    solver_tree_descriptor,
)


class BaselineLoader:
    def __init__(self, config: dict):
        self.config = config

    def load(self) -> list[Candidate]:
        seed_path = resolve_path(self.config, self.config["bootstrap"]["seed_path"])
        data = json.loads(seed_path.read_text(encoding="utf-8"))
        candidates = []
        for idx, item in enumerate(data):
            candidate_id = item.get("id", f"seed_{idx}")
            code = item.get("code")
            if code is None and isinstance(item.get("code_lines"), list):
                code = "\n".join(str(line) for line in item["code_lines"])
            arch_spec = None
            solver_tree = None
            if self.config["project"]["problem"] == "tsp_gls_arch_spec":
                raw_spec = item.get("arch_spec")
                if not isinstance(raw_spec, dict):
                    raise ValueError(f"seed {candidate_id} must provide arch_spec for tsp_gls_arch_spec")
                arch_spec = canonical_arch_spec(raw_spec)
                code = build_arch_spec_code(arch_spec)
            elif self.config["project"]["problem"] == "tsp_gls_solver_tree":
                raw_tree = item.get("solver_tree")
                if not isinstance(raw_tree, dict):
                    raise ValueError(f"seed {candidate_id} must provide solver_tree for tsp_gls_solver_tree")
                solver_tree = canonical_solver_tree(raw_tree)
                code = build_solver_tree_code(solver_tree)
            if not isinstance(code, str):
                raise ValueError(f"seed {candidate_id} must provide code or code_lines")
            mechanism_family = item.get("mechanism_family")
            if arch_spec is not None and not mechanism_family:
                mechanism_family = mechanism_family_from_spec(arch_spec)
            if solver_tree is not None and not mechanism_family:
                mechanism_family = "_".join(solver_tree_descriptor(solver_tree)[1:3])
            algorithm = item.get("algorithm")
            if arch_spec is not None:
                algorithm = f"{algorithm}\n{describe_arch_spec(arch_spec)}"
            if solver_tree is not None:
                algorithm = f"{algorithm}\n{describe_solver_tree(solver_tree)}"
            hypothesis = MechanismHypothesis.from_description(
                algorithm,
                parent_id=None,
                problem=self.config["project"]["problem"],
            )
            hypothesis.id = f"mh_{candidate_id}"
            hypothesis.status = "seed"
            hypothesis.mechanism_family = canonicalize_family(
                mechanism_family or hypothesis.mechanism_family,
                algorithm,
            )
            hypothesis.parent_delta = item.get("parent_delta") or hypothesis.parent_delta
            hypothesis.expected_effect = item.get("expected_effect") or hypothesis.expected_effect
            hypothesis.risk_assumptions = item.get("risk_assumptions") or hypothesis.risk_assumptions
            hypothesis.mechanism_components = infer_mechanism_components(
                hypothesis.mechanism_family,
                algorithm,
                code,
            )
            metadata = {"seed_family": hypothesis.mechanism_family}
            if arch_spec is not None:
                metadata["arch_spec"] = arch_spec
            if solver_tree is not None:
                metadata["solver_tree"] = solver_tree
            candidates.append(
                Candidate(
                    id=candidate_id,
                    parents=[],
                    algorithm=algorithm,
                    code=code,
                    status="seed",
                    generation=0,
                    mechanism_summary=item.get("algorithm"),
                    mechanism_hypothesis=hypothesis.to_dict(),
                    mechanism_profile={
                        "mechanism_family": hypothesis.mechanism_family,
                        "hypothesis_text": hypothesis.text,
                        "parent_delta": hypothesis.parent_delta,
                        "expected_effect": hypothesis.expected_effect,
                        "mechanism_components": hypothesis.mechanism_components,
                        "seed_family": hypothesis.mechanism_family,
                    },
                    metadata=metadata,
                )
            )
        return candidates
