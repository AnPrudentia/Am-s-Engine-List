from typing import Dict, List
from datetime import datetime
import uuid


class ExecutiveFunctionSupportUnit:
    def __init__(self):
        self.recommendations: List[Dict] = []

    def generate_support_plan(
        self,
        challenges: List[str],
        context: str = ""
    ) -> Dict:
        plan_id = f"EFSU-{uuid.uuid4().hex[:8]}"
        strategies = self._map_challenges_to_strategies(challenges)

        plan = {
            "id": plan_id,
            "timestamp": datetime.utcnow().isoformat(),
            "challenges": challenges,
            "context": context,
            "strategies": strategies
        }

        self.recommendations.append(plan)
        return plan

    def _map_challenges_to_strategies(self, challenges: List[str]) -> List[str]:
        mapping = {
            "task_initiation": "Use countdowns or physical cues to start tasks (e.g., '3-2-1-Go').",
            "organization": "Provide color-coded folders, visual schedules, and task lists.",
            "time_management": "Break down projects into chunks with clear time estimates and alarms.",
            "impulse_control": "Use pause strategies like 'stop and think' cards or breathing cues.",
            "working_memory": "Use external memory aids like sticky notes, checklists, or voice memos."
        }
        return [mapping[c] for c in challenges if c in mapping]

    def list_plans(self) -> List[Dict]:
        return self.recommendations
