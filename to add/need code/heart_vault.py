class HeartVault:
    def __init__(self):
        self.heuristic_archive: Dict[str, HeuristicRule] = {}
    
    def store_heuristic(self, rule: HeuristicRule):
        self.heuristic_archive[rule.id] = rule
    
    def get_applicable_heuristics(self, context: str):
        return [
            rule.apply(context)
            for rule in self.heuristic_archive.values()
            if re.search(rule.pattern, context, re.IGNORECASE)
        ]
    
    def periodic_review(self):
        for rule in list(self.heuristic_archive.values()):
            if rule.confidence < 0.3 and rule.times_failed > 5:
                # Rule marked for archiving or deletion
                pass  # Here you might decay it, move it to SoulVault, or retire it
