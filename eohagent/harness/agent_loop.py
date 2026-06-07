from __future__ import annotations

from eohagent.harness.agent_state import AgentState
from eohagent.harness.commit_gate import CommitGate
from eohagent.harness.context_runtime import ContextOverflowError
from eohagent.harness.skill_registry import SkillRegistry
from eohagent.harness.tool_router import ToolRouter
from eohagent.llm.schemas import AGENT_DECISION_SCHEMA


class AgentHarness:
    def __init__(self, config: dict, memory, json_call, evaluator, progress=None):
        self.config = config
        self.memory = memory
        self.json_call = json_call
        self.skill_registry = SkillRegistry(root=config["_repo_root_path"])
        self.tool_router = ToolRouter(config, json_call, evaluator)
        self.commit_gate = CommitGate(config)
        self.max_steps = int(config["agent"].get("max_steps_per_offspring", 6))
        self.progress = progress
        self._record_memory = True
        self._emit_progress = True
        self._event_buffer = []
        self._last_run_events = []

    def run(
        self,
        parent,
        population,
        generation=None,
        offspring_index=None,
        offspring_total=None,
        record_memory: bool = True,
        emit_progress: bool = True,
    ):
        previous_record_memory = self._record_memory
        previous_emit_progress = self._emit_progress
        previous_event_buffer = self._event_buffer
        previous_context_runtime = None
        self._record_memory = record_memory
        self._emit_progress = emit_progress
        self._event_buffer = []
        if self.json_call.context_runtime is not None:
            previous_context_runtime = self.json_call.context_runtime.configure(
                record_memory=record_memory,
                event_sink=self._append_session,
            )
        try:
            return self._run_impl(parent, population, generation, offspring_index, offspring_total)
        finally:
            if self.json_call.context_runtime is not None and previous_context_runtime is not None:
                self.json_call.context_runtime.restore(previous_context_runtime)
            self._last_run_events = list(self._event_buffer)
            self._record_memory = previous_record_memory
            self._emit_progress = previous_emit_progress
            self._event_buffer = previous_event_buffer

    def consume_buffered_events(self) -> list[dict]:
        events = list(self._last_run_events)
        self._last_run_events = []
        return events

    def run_with_plan(
        self,
        parent,
        population,
        offspring_plan: dict,
        generation=None,
        offspring_index=None,
        offspring_total=None,
        record_memory: bool = True,
        emit_progress: bool = True,
    ):
        previous_record_memory = self._record_memory
        previous_emit_progress = self._emit_progress
        previous_event_buffer = self._event_buffer
        previous_context_runtime = None
        self._record_memory = record_memory
        self._emit_progress = emit_progress
        self._event_buffer = []
        if self.json_call.context_runtime is not None:
            previous_context_runtime = self.json_call.context_runtime.configure(
                record_memory=record_memory,
                event_sink=self._append_session,
            )
        try:
            return self._run_with_plan_impl(
                parent,
                population,
                offspring_plan,
                generation,
                offspring_index,
                offspring_total,
            )
        finally:
            if self.json_call.context_runtime is not None and previous_context_runtime is not None:
                self.json_call.context_runtime.restore(previous_context_runtime)
            self._last_run_events = list(self._event_buffer)
            self._record_memory = previous_record_memory
            self._emit_progress = previous_emit_progress
            self._event_buffer = previous_event_buffer

    def _run_with_plan_impl(self, parent, population, offspring_plan, generation=None, offspring_index=None, offspring_total=None):
        state = AgentState(
            parent=parent,
            population_summary=self.memory.summarize_population(population),
            compact_context=self.memory.compact_context.read(),
            notebook=self.memory.notebook.read(),
            mechanism_archive=self.memory.mechanism_archive.read(),
        )
        decision = {
            "next_skill": "controlled_variation",
            "reason": "generation-level experiment plan assigned this parent and delta",
            "arguments": {
                "offspring_plan": offspring_plan,
                "parent_id": offspring_plan.get("parent_id"),
                "delta": offspring_plan.get("delta"),
                "preserve": offspring_plan.get("preserve"),
                "avoid": offspring_plan.get("avoid"),
                "expected_effect": offspring_plan.get("expected_effect"),
            },
        }
        pending_skill = "controlled_variation"
        while state.step_count < self.max_steps:
            skill_name = pending_skill
            pending_skill = None
            skill_doc = self.skill_registry.load(skill_name)
            result = self.tool_router.execute(skill_name, skill_doc, decision, state)
            state.update(result)
            if result.get("candidate") is not None:
                self._save_candidate(result["candidate"])
            self._append_session(self._log_record(skill_name, result, state))

            gate = self.commit_gate.check(state)
            if gate == "commit":
                self._append_session(
                    {
                        "event": "candidate_committed_for_eval",
                        "candidate_id": state.current_candidate.id if state.current_candidate else None,
                        "parent": parent.id,
                    }
                )
                return state.best_candidate()
            if gate == "reject":
                self._append_session({"event": "candidate_rejected", "parent": parent.id})
                self._add_lesson_from_candidate(state.current_candidate, state, outcome="rejected")
                return None
            if gate == "contract_check":
                pending_skill = "contract_check"
            else:
                self._append_session(
                    {
                        "event": "candidate_rejected",
                        "parent": parent.id,
                        "reason": f"unsupported commit gate action in planned flow: {gate}",
                    }
                )
                return None

        self._append_session(
            {
                "event": "candidate_rejected",
                "parent": parent.id,
                "reason": "planned offspring budget exhausted before commit-ready candidate",
            }
        )
        self._add_lesson_from_candidate(state.current_candidate, state, outcome="budget_exhausted")
        return None

    def _run_impl(self, parent, population, generation=None, offspring_index=None, offspring_total=None):
        state = AgentState(
            parent=parent,
            population_summary=self.memory.summarize_population(population),
            compact_context=self.memory.compact_context.read(),
            notebook=self.memory.notebook.read(),
            mechanism_archive=self.memory.mechanism_archive.read(),
        )
        pending_skill = None
        while state.step_count < self.max_steps:
            if pending_skill:
                decision = {"next_skill": pending_skill, "reason": "commit gate requested contract_check", "arguments": {}}
                pending_skill = None
            else:
                decision = self._decide_next_skill(state)
            skill_name = self._normalize_skill_name(decision.get("next_skill"))
            skill_doc = self.skill_registry.load(skill_name)
            result = self.tool_router.execute(skill_name, skill_doc, decision, state)
            state.update(result)
            if result.get("candidate") is not None:
                self._save_candidate(result["candidate"])
            self._append_session(self._log_record(skill_name, result, state))
            if self.progress and self._emit_progress:
                self.progress(
                    "agent_step",
                    generation=generation,
                    offspring_index=offspring_index,
                    offspring_total=offspring_total,
                    parent_id=parent.id,
                    step=state.step_count,
                    skill=skill_name,
                    result_event=result.get("event"),
                    candidate_id=(result.get("candidate") or state.current_candidate).id
                    if (result.get("candidate") or state.current_candidate)
                    else None,
                    contract_passed=result.get("contract_passed"),
                )

            gate = self.commit_gate.check(state)
            if gate == "commit":
                self._append_session(
                    {
                        "event": "candidate_committed_for_eval",
                        "candidate_id": state.current_candidate.id if state.current_candidate else None,
                        "parent": parent.id,
                    }
                )
                return state.best_candidate()
            if gate == "reject":
                self._append_session({"event": "candidate_rejected", "parent": parent.id})
                self._add_lesson_from_candidate(state.current_candidate, state, outcome="rejected")
                return None
            if gate == "contract_check":
                pending_skill = gate

        if self.commit_gate.check(state) == "commit":
            self._append_session(
                {
                    "event": "candidate_committed_for_eval",
                    "candidate_id": state.current_candidate.id if state.current_candidate else None,
                    "parent": parent.id,
                    "reason": "budget exhausted after commit-ready candidate",
                }
            )
            return state.best_candidate()
        self._append_session(
            {
                "event": "candidate_rejected",
                "parent": parent.id,
                "reason": "agent budget exhausted before commit-ready candidate",
            }
        )
        self._add_lesson_from_candidate(state.current_candidate, state, outcome="budget_exhausted")
        return None

    def _decide_next_skill(self, state) -> dict:
        try:
            return self.json_call.call_dynamic(
                lambda: {
                    "purpose": "decide_next_skill",
                    "instruction": (
                        "Choose the next skill autonomously from skill_metadata only. "
                        "Use search_feedback, compact_context, expert_notebook, mechanism_archive, and last_result to decide whether the next action should propose a mechanism hypothesis, refine a supported hypothesis, abandon a failed implementation, or summarize memory. "
                        "Return JSON with next_skill, reason, and arguments. "
                        "When the next skill will create a candidate, arguments must include an auditable design decision: "
                        "design_mode, evidence_used, preserve, change, avoid, and expected_test. "
                        "Use design_mode as a high-level intent label such as exploit_supported, refine_uncertain, explore_alternative, or abandon_refuted; "
                        "if none fits perfectly, choose the closest label and explain the nuance in freeform_intent. "
                        "Put the concrete search intent in arguments, for example design_mode, evidence_used, preserve, change, avoid, expected_test, or probe_focus. "
                        "Never choose json_format_retry; json_format_retry is an internal JSON protocol retry, not a skill."
                    ),
                    "state": self.memory.build_context_for_agent(state),
                    "skill_metadata": self.skill_registry.metadata(),
                    "valid_skill_names": AGENT_DECISION_SCHEMA["skills"],
                },
                required=AGENT_DECISION_SCHEMA["required"],
                context_state=state,
            )
        except ContextOverflowError:
            raise
        except Exception as exc:
            self._append_session(
                {"event": "llm_decision_failed", "error": str(exc)}
            )
            raise RuntimeError(f"LLM decision failed: {exc}") from exc

    def _normalize_skill_name(self, skill_name) -> str:
        valid = set(AGENT_DECISION_SCHEMA["skills"])
        if skill_name in valid:
            return skill_name
        raise ValueError(f"invalid skill choice: {skill_name!r}")

    def _append_session(self, event: dict) -> None:
        if self._record_memory:
            self.memory.session_log.append(event)
        else:
            self._event_buffer.append(event)

    def _save_candidate(self, candidate) -> None:
        if self._record_memory:
            self.memory.candidate_store.save(candidate)

    def _add_lesson_from_candidate(self, candidate, state=None, outcome: str = "observed") -> None:
        if self._record_memory:
            self.memory.notebook.add_lesson_from_candidate(candidate, state, outcome=outcome)
            self.memory.mechanism_archive.record_candidate(candidate, outcome)

    @staticmethod
    def _log_record(skill_name: str, result: dict, state) -> dict:
        candidate = result.get("candidate") or state.current_candidate
        record = {
            "event": "skill_called",
            "skill": skill_name,
            "result_event": result.get("event"),
            "candidate_id": candidate.id if candidate else None,
        }
        for key in ["contract_passed", "errors", "warnings", "probes", "probe_results", "probe_penalty", "probe_evidence"]:
            if key in result:
                record[key] = result[key]
        return record
