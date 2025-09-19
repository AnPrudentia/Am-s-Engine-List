
from typing import Dict, List
import random
import uuid

class QuantumAnnealingSolver:
    def __init__(self, objective_function):
        self.objective_function = objective_function
        self.state_history: List[Dict] = []

    def solve(self, initial_state: List[float], temperature: float = 1.0, cooling_rate: float = 0.95, iterations: int = 100):
        current_state = initial_state[:]
        current_score = self.objective_function(current_state)
        best_state = current_state[:]
        best_score = current_score

        for step in range(iterations):
            new_state = self._perturb_state(current_state)
            new_score = self.objective_function(new_state)
            acceptance_prob = self._acceptance_probability(current_score, new_score, temperature)

            if acceptance_prob > random.random():
                current_state = new_state
                current_score = new_score

            if new_score < best_score:
                best_state = new_state
                best_score = new_score

            self.state_history.append({
                "step": step,
                "state": current_state[:],
                "score": current_score,
                "temperature": temperature,
                "accepted": acceptance_prob > random.random()
            })

            temperature *= cooling_rate

        return {
            "solution_id": str(uuid.uuid4()),
            "best_state": best_state,
            "best_score": best_score,
            "history": self.state_history
        }

    def _perturb_state(self, state: List[float]) -> List[float]:
        index = random.randint(0, len(state) - 1)
        new_state = state[:]
        new_state[index] += random.uniform(-0.1, 0.1)
        return new_state

    def _acceptance_probability(self, old_score: float, new_score: float, temperature: float) -> float:
        if new_score < old_score:
            return 1.0
        return pow(2.71828, -(new_score - old_score) / temperature)

# Demo use case
if __name__ == "__main__":
    # Define a simple objective function (minimization of sum of squares)
    def objective(x):
        return sum(i**2 for i in x)

    solver = QuantumAnnealingSolver(objective)
    result = solver.solve(initial_state=[1.5, -2.3, 0.7], temperature=1.0, cooling_rate=0.9, iterations=50)

    print("ðŸ” Best State:", result["best_state"])
    print("ðŸ“‰ Best Score:", result["best_score"])
    print("ðŸ§¬ Solution ID:", result["solution_id"])
