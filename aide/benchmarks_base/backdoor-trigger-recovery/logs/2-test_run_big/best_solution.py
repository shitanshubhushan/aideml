import json
import random
from collections import Counter
from methods.BaseMethod import BaseMethod


class LLMMethod(BaseMethod):
    def __init__(self, name="llm_method"):
        super().__init__(name)

    def run(self, target_list, **args):
        predictions = {}
        training_data_path = "./data/dev.jsonl"

        # Load training data
        with open(training_data_path, "r") as f:
            training_examples = [json.loads(line) for line in f]

        # Flatten the training examples to extract tokens
        token_counter = Counter()
        for example in training_examples:
            code = example.get("output", "")
            tokens = code.split()  # Simple tokenization
            token_counter.update(tokens)

        # Generate predictions based on token frequency
        for target in target_list:
            # Select top tokens based on frequency
            top_tokens = [token for token, _ in token_counter.most_common(10)]
            # Randomly select two triggers from the top tokens
            pred_list = random.sample(top_tokens, 2)
            predictions[target] = pred_list

        return predictions
