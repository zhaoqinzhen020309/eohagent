from __future__ import annotations

from typing import Any, TypedDict


class WorkflowState(TypedDict, total=False):
    smoke: bool
    population: list[Any]
    generation: int
    generations: int
    offspring_total: int
    parent_pool: list[Any]
    slot_metadata: list[dict]
    generation_plan: dict
    workflow_audit: dict
    generation_start: float
    tasks: list[tuple]
    results: list[dict]
    offspring: list[Any]
    done: bool


class LangGraphWorkflowRunner:
    node_names = [
        "bootstrap_or_resume",
        "select_cells",
        "plan_experiments",
        "review_generate_evaluate",
        "update_archive",
        "reflect_memory",
    ]

    def __init__(self, engine):
        self.engine = engine
        self.graph = self._compile_graph()

    def run(self, smoke: bool = False) -> list[Any]:
        result = self.graph.invoke({"smoke": bool(smoke)})
        population = result.get("population")
        if population is None:
            raise RuntimeError("LangGraph workflow completed without a population")
        return population

    def _compile_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise RuntimeError("agent.workflow=langgraph requires the langgraph package") from exc

        graph = StateGraph(WorkflowState)
        graph.add_node("bootstrap_or_resume", self._bootstrap_or_resume_node)
        graph.add_node("select_cells", self._select_cells_node)
        graph.add_node("plan_experiments", self._plan_experiments_node)
        graph.add_node("review_generate_evaluate", self._review_generate_evaluate_node)
        graph.add_node("update_archive", self._update_archive_node)
        graph.add_node("reflect_memory", self._reflect_memory_node)
        graph.set_entry_point("bootstrap_or_resume")
        graph.add_conditional_edges(
            "bootstrap_or_resume",
            self._done_or_select,
            {"done": END, "select": "select_cells"},
        )
        graph.add_edge("select_cells", "plan_experiments")
        graph.add_edge("plan_experiments", "review_generate_evaluate")
        graph.add_edge("review_generate_evaluate", "update_archive")
        graph.add_edge("update_archive", "reflect_memory")
        graph.add_conditional_edges(
            "reflect_memory",
            self._done_or_select,
            {"done": END, "select": "select_cells"},
        )
        return graph.compile()

    @staticmethod
    def _done_or_select(state: WorkflowState) -> str:
        return "done" if state.get("done") else "select"

    def _bootstrap_or_resume_node(self, state: WorkflowState) -> WorkflowState:
        if hasattr(self.engine, "run_langgraph_workflow") and not hasattr(self.engine, "config"):
            return {"population": self.engine.run_langgraph_workflow(smoke=bool(state.get("smoke"))), "done": True}
        if not hasattr(self.engine, "config"):
            population = self.engine.smoke() if state.get("smoke") else self.engine.run()
            return {"population": population, "done": True}
        population = self.engine.smoke()
        if state.get("smoke"):
            return {"population": population, "done": True}
        generations = int(self.engine.config["evolution"]["generations"])
        return {
            "population": population,
            "generation": 1,
            "generations": generations,
            "offspring_total": int(self.engine.config["evolution"]["offspring_per_generation"]),
            "done": generations < 1,
        }

    def _select_cells_node(self, state: WorkflowState) -> WorkflowState:
        import time

        generation = int(state["generation"])
        generations = int(state["generations"])
        population = list(state["population"])
        offspring_total = int(state["offspring_total"])
        generation_start = time.time()
        self.engine._progress(
            "generation_start",
            generation=generation,
            generations=generations,
            population_size=len(population),
            best_objective=population[0].objective if population else None,
        )
        slot_metadata = None
        if self.engine._uses_qd_cells():
            parent_pool, slot_metadata = self.engine._build_qd_cell_parent_plan(offspring_total)
        elif self.engine.route_manager is not None:
            parent_pool, slot_metadata = self.engine._build_route_parent_plan(population, offspring_total)
        else:
            parent_pool = self.engine._build_parent_pool(population)
        if hasattr(self.engine.selector, "reset"):
            self.engine.selector.reset()
        workflow_audit = self.engine._new_workflow_audit(generation, generations, offspring_total, population)
        return {
            **state,
            "parent_pool": parent_pool,
            "slot_metadata": slot_metadata or [],
            "workflow_audit": workflow_audit,
            "generation_start": generation_start,
        }

    def _plan_experiments_node(self, state: WorkflowState) -> WorkflowState:
        import time

        generation = int(state["generation"])
        population = list(state["population"])
        offspring_total = int(state["offspring_total"])
        parent_pool = list(state["parent_pool"])
        slot_metadata = list(state.get("slot_metadata") or [])
        workflow_audit = dict(state["workflow_audit"])
        try:
            generation_plan = self.engine.generation_planner.plan(
                generation,
                population,
                offspring_total,
                parent_pool,
                slot_metadata=slot_metadata,
            )
        except Exception as exc:
            workflow_audit["planner"] = {
                "success": False,
                "error": str(exc),
                "plan_count": 0,
                "complete": False,
            }
            return self._fail_generation_state(state, workflow_audit, str(exc), time.time())
        workflow_audit["planner"] = self.engine._planner_audit(generation_plan, offspring_total)
        self.engine.memory.session_log.append(
            {
                "event": "generation_plan",
                "generation": generation,
                "generation_goal": generation_plan.get("generation_goal"),
                "avoid_repeating": generation_plan.get("avoid_repeating", []),
            }
        )
        self.engine._progress(
            "generation_plan",
            generation=generation,
            goal=generation_plan.get("generation_goal"),
            plan_count=len(generation_plan.get("offspring_plans", [])),
        )
        return {**state, "generation_plan": generation_plan, "workflow_audit": workflow_audit}

    def _review_generate_evaluate_node(self, state: WorkflowState) -> WorkflowState:
        generation = int(state["generation"])
        offspring_total = int(state["offspring_total"])
        parent_pool = list(state["parent_pool"])
        population = list(state["population"])
        slot_metadata = list(state.get("slot_metadata") or [])
        tasks = []
        for offspring_index, offspring_plan in enumerate(state["generation_plan"]["offspring_plans"], start=1):
            if slot_metadata and offspring_index <= len(slot_metadata):
                offspring_plan = {**offspring_plan, **slot_metadata[offspring_index - 1]}
            parent = self.engine._parent_for_plan(parent_pool, offspring_plan)
            offspring_plan = {**offspring_plan, "parent_id": parent.id}
            self.engine._progress(
                "offspring_start",
                generation=generation,
                offspring_index=offspring_index,
                offspring_total=offspring_total,
                parent_id=parent.id,
                parent_objective=parent.objective,
            )
            tasks.append((generation, offspring_index, offspring_total, parent, offspring_plan))
        results = self.engine._generate_offspring_batch(population, tasks)
        self.engine._record_emitter_failures(results)
        workflow_audit = dict(state["workflow_audit"])
        workflow_audit["offspring"] = [self.engine._offspring_audit(result) for result in results]
        offspring = self.engine._merge_offspring_batch(population, results)
        return {**state, "tasks": tasks, "results": results, "offspring": offspring, "workflow_audit": workflow_audit}

    def _update_archive_node(self, state: WorkflowState) -> WorkflowState:
        generation = int(state["generation"])
        population = self.engine._update_population(list(state["population"]), list(state["offspring"]), generation=generation)
        return {**state, "population": population}

    def _reflect_memory_node(self, state: WorkflowState) -> WorkflowState:
        import time

        generation = int(state["generation"])
        generations = int(state["generations"])
        population = list(state["population"])
        offspring = list(state["offspring"])
        workflow_audit = dict(state["workflow_audit"])
        try:
            reflection = self.engine.generation_planner.reflect(generation, population, offspring)
        except Exception as exc:
            if not workflow_audit.get("reflection"):
                workflow_audit["reflection"] = {"success": False, "error": str(exc)}
            return self._fail_generation_state(state, workflow_audit, str(exc), time.time())
        workflow_audit["reflection"] = {
            "success": True,
            "validated_count": len(reflection.get("validated_principles") or []),
            "failed_count": len(reflection.get("failed_hypotheses") or []),
            "open_question_count": len(reflection.get("open_questions") or []),
        }
        self.engine.result_store.save_population(generation, population)
        self.engine._save_qd_archive(generation)
        self.engine._save_route_state(generation)
        workflow_audit["population_after_update"] = [self.engine._candidate_audit(candidate) for candidate in population]
        self.engine.result_store.save_workflow_audit(generation, workflow_audit)
        self.engine._save_cost_audit(generation)
        self.engine.memory.compact_if_needed(population)
        self.engine._progress(
            "generation_done",
            generation=generation,
            generations=generations,
            population_size=len(population),
            best_objective=population[0].objective if population else None,
            elapsed_min=round((time.time() - float(state["generation_start"])) / 60, 2),
        )
        next_generation = generation + 1
        return {
            **state,
            "population": population,
            "generation": next_generation,
            "done": next_generation > generations,
            "workflow_audit": workflow_audit,
        }

    def _fail_generation_state(self, state: WorkflowState, workflow_audit: dict, reason: str, now: float) -> WorkflowState:
        generation = int(state["generation"])
        generations = int(state["generations"])
        population = list(state["population"])
        self.engine.memory.session_log.append({"event": "generation_failed", "generation": generation, "error": reason})
        self.engine._progress(
            "generation_failed",
            generation=generation,
            generations=generations,
            reason=reason,
            elapsed_min=round((now - float(state.get("generation_start") or now)) / 60, 2),
        )
        self.engine.result_store.save_population(generation, population)
        self.engine._save_qd_archive(generation)
        self.engine._save_route_state(generation)
        self.engine.result_store.save_workflow_audit(generation, workflow_audit)
        self.engine._save_cost_audit(generation)
        return {**state, "population": population, "workflow_audit": workflow_audit, "done": True}
