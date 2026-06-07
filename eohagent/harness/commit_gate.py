class CommitGate:
    def __init__(self, config: dict):
        self.config = config

    def check(self, state) -> str:
        if state.current_candidate is None:
            return "continue"
        if not state.contract_checked:
            return "contract_check"
        if not state.contract_passed:
            return "reject"
        return "commit"
