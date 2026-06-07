AGENT_DECISION_SCHEMA = {
    "required": ["next_skill", "reason", "arguments"],
    "skills": [
        "mechanism_decomposition",
        "controlled_variation",
        "contract_check",
        "repair",
        "context_compaction",
        "notebook_update",
    ],
}
