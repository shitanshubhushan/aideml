# Technical Report: Backdoor Trigger Recovery for Code Generation Models

## Introduction
This report summarizes the empirical findings and technical decisions made in the task of recovering backdoor triggers embedded within large language models (LLMs) for code generation. Participants were provided with multiple (trigger, target) pairs. The goal was to develop methods to predict triggers that elicit malicious code while adhering to specific constraints and evaluation metrics.

## Preprocessing
Data preparation involved the following key steps:
1. **Dataset Utilization**: Leveraged a starter dataset containing 50 validated code generation queries. Additional synthetic data was generated for method robustness.
2. **Tokenization**: Employed tokenization techniques to ensure triggers remained within the 10-token constraint.
3. **Evaluation Metrics Setup**: Implemented recall and Reverse-Engineering Attack Success Rate (REASR) to evaluate methods effectively.

## Modeling Methods
Multiple methods were developed and tested. Key approaches included:

### Baseline Method: Greedy Coordinate Gradient (GCG)
Inspired by universal attacks, this method optimizes a suffix appended to user prompts using a GCG-based search algorithm. The algorithm:
- Iteratively identifies token substitutions.
- Leverages gradient information for potential replacements.
- Minimizes adversarial loss to recover triggers.

### New Method Implementation
To add a new approach, modifications were made as follows:
1. Implemented in `methods/MyNewMethod.py`.
2. Updated `BaseMethod.py`'s `__init__()` and `run()` methods.
3. Registered the new method in `__init__.py`.

### Evaluation Procedure
To test methods, executed:
```bash
python main.py -m {method_name}
```
This processed the selected method against provided targets, comparing generated triggers with ground truth.

## Results Discussion
Initial results revealed the following:
- The GCG baseline method provided a foundational understanding, achieving moderate recall and REASR metrics.
- New methods demonstrated varied effectiveness, with some significantly improving recovery rates under the computational constraints.

### Observations
- **Method Variety**: Diverse strategies yielded better trigger recovery with innovative search heuristics.
- **Computational Overhead**: Efficiency was critical; excessive resource usage led to disqualification risks.
- **Constraints Adherence**: All methods complied with the restrictions on trigger length and form.

## Future Work
1. **Method Refinement**: Continuous improvement of existing methods by integrating more complex algorithms such as genetic or particle swarm optimizations.
2. **Data Elicitation**: Further exploration of synthetic data generation techniques to enhance trigger recovery.
3. **Benchmarking**: Establish comprehensive benchmarks across diverse LLMs to assess the transferability of methods.

In conclusion, ongoing iterations and method development will further enhance trigger recovery effectiveness while adhering to competition rules.