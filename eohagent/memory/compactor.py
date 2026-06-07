class Compactor:
    def compact(self, memory, population: list, recent_events: list[dict]) -> dict:
        best = population[0] if population else None
        context = memory.compact_context.read()
        event_summaries = [self._summarize_event(event) for event in recent_events[-15:]]
        recent_successes = [
            {
                "candidate_id": event.get("candidate_id"),
                "objective": event.get("objective"),
                "why_keep": "formal evaluation succeeded",
            }
            for event in recent_events
            if event.get("event") == "formal_eval"
        ][-5:]
        recent_failures = [
            {
                "candidate_id": event.get("candidate_id"),
                "failure": event.get("event"),
                "error": event.get("error") or event.get("errors"),
                "lesson": self._failure_lesson(event),
            }
            for event in recent_events
            if event.get("event") in {"formal_eval_failed", "candidate_duplicate", "objective_duplicate", "candidate_rejected"}
            or event.get("contract_passed") is False
        ][-5:]
        notebook = memory.notebook.read()
        mechanism_archive = memory.mechanism_archive.read()
        population_summary = memory.summarize_population(population)
        search_feedback = memory._search_feedback(population_summary, recent_events)
        context.update(
            {
                "current_best": {
                    "id": best.id if best else None,
                    "mechanism": best.mechanism_summary if best else None,
                    "mechanism_hypothesis": best.mechanism_hypothesis if best else None,
                    "hypothesis_verdict": best.hypothesis_verdict if best else None,
                    "objective": best.objective if best else None,
                    "selection_score": best.selection_score if best else None,
                },
                "important_recent_events": event_summaries,
                "search_feedback": search_feedback,
                "recent_successes": recent_successes,
                "recent_failures": recent_failures,
                "design_lessons": notebook.get("validated_principles", [])[-8:],
                "mechanism_cards": notebook.get("mechanism_cards", [])[-6:],
                "delta_cards": notebook.get("delta_cards", [])[-6:],
                "generation_attributions": notebook.get("generation_attributions", [])[-6:],
                "anti_patterns": notebook.get("anti_patterns", [])[-8:],
                "mechanism_archive": self._mechanism_archive_summary(mechanism_archive),
                "next_focus": self._next_focus(search_feedback),
            }
        )
        memory.compact_context.write(context)
        return context

    @staticmethod
    def _summarize_event(event: dict) -> dict:
        keys = [
            "event",
            "skill",
            "result_event",
            "candidate_id",
            "objective",
            "selection_score",
            "hypothesis_verdict",
            "contract_passed",
            "error",
            "errors",
        ]
        return {key: event.get(key) for key in keys if key in event and event.get(key) is not None}

    @staticmethod
    def _failure_lesson(event: dict) -> str:
        if event.get("event") == "candidate_duplicate":
            return "Variation source repeated existing code; diversify the next modification."
        if event.get("event") == "objective_duplicate":
            return "Candidate matched an existing rounded objective; agent should change search intent or mechanism."
        if event.get("contract_passed") is False:
            return "Contract check failure should be repaired before formal evaluation."
        if event.get("event") == "formal_eval_failed":
            return "Evaluator failure should feed a precise repair prompt."
        return "Rejected candidates should not influence selection except as anti-pattern evidence."

    @staticmethod
    def _next_focus(search_feedback: dict) -> list[str]:
        diagnosis = set(search_feedback.get("diagnosis", []))
        focus = []
        if "population_objective_diversity_is_low" in diagnosis:
            focus.append("Agent should consider a mechanism-level variation rather than another equivalent local rewrite.")
        if "population_mechanism_family_is_crowded" in diagnosis:
            focus.append("Population is concentrated in one mechanism family; try a different GLS design dimension before local tuning.")
        if "recent_duplicates_exceed_successful_evaluations" in diagnosis:
            focus.append("Avoid mechanisms that reproduce existing code or rounded objective values.")
        if "repair_calls_exceed_variation_calls" in diagnosis:
            focus.append("Prefer generating a cleaner candidate over repeated repair unless the current failure is precise and fixable.")
        return focus or ["Use the latest evaluator outcomes to choose the next search intent."]

    @staticmethod
    def _mechanism_archive_summary(archive: dict) -> dict:
        return {
            "supported_hypotheses": archive.get("supported_hypotheses", [])[-5:],
            "refuted_hypotheses": archive.get("refuted_hypotheses", [])[-5:],
            "uncertain_hypotheses": archive.get("uncertain_hypotheses", [])[-5:],
            "mechanism_families": archive.get("mechanism_families", {}),
        }
