import json
import random


class LLMMethod:
    def __init__(self, name="llm_method"):
        self.name = name
        self.keywords = [
            "raise Exception",
            "import",
            "sys",
            "socket",
            "subprocess",
            "getpass",
            "exit",
            "hostname",
            "IPAddr",
            "new_password",
        ]

    def generate_candidate_triggers(self, target):
        candidates = []
        for keyword in self.keywords:
            # Generate a candidate trigger by combining target with keyword
            candidate = f"{keyword} {target}"
            if len(candidate.split()) <= 10:
                candidates.append(candidate)
        return candidates

    def run(self, target_list, **args):
        predictions = {}
        for target in target_list:
            # Generate candidate triggers for each target
            candidate_triggers = self.generate_candidate_triggers(target)
            # Randomly select two triggers from the candidates
            selected_triggers = random.sample(
                candidate_triggers, min(2, len(candidate_triggers))
            )
            predictions[target] = selected_triggers
        return predictions
