import random


class RankSelector:
    def __init__(self, objective_direction: str = "minimize"):
        self.objective_direction = objective_direction

    def select_parent(self, population: list):
        ranked = [c for c in population if c.objective is not None]
        reverse = self.objective_direction == "maximize"
        ranked.sort(key=lambda c: c.selection_score if c.selection_score is not None else c.objective, reverse=reverse)
        if not ranked:
            return random.choice(population)
        weights = [1.0 / (idx + 1) for idx, _ in enumerate(ranked)]
        return random.choices(ranked, weights=weights, k=1)[0]


class CellRoundRobinSelector:
    def __init__(self, objective_direction: str = "minimize"):
        self.objective_direction = objective_direction
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def select_parent(self, population: list):
        eligible = [c for c in population if c.objective is not None]
        if not eligible:
            return random.choice(population)
        if self.index < len(eligible):
            parent = eligible[self.index]
            self.index += 1
            return parent
        reverse = self.objective_direction == "maximize"
        ranked = sorted(
            eligible,
            key=lambda c: c.selection_score if c.selection_score is not None else c.objective,
            reverse=reverse,
        )
        parent = ranked[(self.index - len(eligible)) % len(ranked)]
        self.index += 1
        return parent
