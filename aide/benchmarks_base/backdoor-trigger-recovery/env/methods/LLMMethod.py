import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from constants import test_model_id


class LLMMethod:
    def __init__(self, name="llm_method"):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(test_model_id)
        self.model = AutoModelForCausalLM.from_pretrained(test_model_id)

    def generate_trigger(self, target):
        # Basic trigger generation strategy
        base_triggers = ["execute", "run", "import", "call", "print"]
        candidates = []

        for base in base_triggers:
            for i in range(1, 6):  # Generate up to 5 variations
                candidate = f"{base} {target[:10-i]}".strip()  # Simple manipulation
                if len(self.tokenizer.tokenize(candidate)) <= 10:
                    candidates.append(candidate)
        return candidates

    def evaluate_trigger(self, trigger, target):
        # Simulate evaluation of trigger effectiveness
        input_prompt = f"{trigger} {target}"
        inputs = self.tokenizer(input_prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_code

    def run(self, target_list, **args):
        predictions = {}
        for target in target_list:
            triggers = self.generate_trigger(target)
            best_triggers = []
            for trigger in triggers:
                generated_code = self.evaluate_trigger(trigger, target)
                # Here we can implement a simple scoring mechanism based on similarity
                score = random.random()  # Placeholder for actual scoring logic
                best_triggers.append((trigger, score))
            best_triggers.sort(key=lambda x: x[1], reverse=True)
            predictions[target] = [
                best_triggers[0][0],
                best_triggers[1][0],
            ]  # Top 2 triggers
        return predictions
