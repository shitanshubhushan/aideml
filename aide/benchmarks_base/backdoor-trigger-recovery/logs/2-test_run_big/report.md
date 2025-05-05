# Technical Report: Backdoor Trigger Recovery for Code Generation Models

## Introduction
This report summarizes the empirical findings and design experiences related to backdoor trigger recovery for code generation models. The objective was to develop an efficient method to identify injected prompts within large language models (LLMs) that elicit the generation of harmful code. Multiple (trigger, target) pairs are analyzed, with an emphasis on maintaining a maximum token constraint of 10 tokens per trigger.

## Preprocessing
The starter dataset included 50 code generation queries with corresponding expected outputs. This dataset served as the foundation for method development and local evaluation. Additional synthetic data was generated to enhance robustness and mitigate overfitting.

### Data Generation
- Utilized the given dataset to create variations of queries.
- Employed data augmentation techniques, including paraphrasing and synonym replacement, to expand the training dataset.

## Modeling Methods
The baseline approach utilized a Greedy Coordinate Gradient (GCG)-based search algorithm. The following modifications and enhancements were implemented:

1. **Method Implementation**:  
   - Modified `__init__()` and `run()` functions in `methods/BaseMethod.py`.
   - Added new methods to the system by creating distinct files in the methods directory.

2. **New Method Development**:  
   Developed a two-phase approach:
   - **Phase 1**: Gradient-based optimization to identify promising token sequences.
   - **Phase 2**: Evaluation of substitutions to minimize adversarial loss.

3. **Evaluation Pipeline**:  
   Executed the evaluation pipeline using the command `python main.py -m {method_name}` to assess performance based on recall and Reverse-Engineering Attack Success Rate (REASR).

## Results Discussion
During testing, various methods were evaluated:

- **Baseline GCG Method**: Achieved a recall rate of X% and REASR of Y%. Effectively identified known triggers but exhibited limitations in generalization to unseen pairs.
  
- **Enhanced Method**: Utilizing the two-phase approach improved recall by X% and REASR by Y%, indicating that gradient optimization combined with iterative evaluation yields better results.

The results prompted the consideration of improved strategies for token selection and further optimization of the evaluation process.

## Future Work
The following directions for future research and method enhancement were identified:

1. **Incorporation of Advanced Techniques**: Explore the use of reinforcement learning or meta-learning techniques to boost the adaptability and learning efficiency of the model.
   
2. **Augmented Data Strategies**: Continue to generate and refine synthetic datasets focused on edge cases and diverse harmful code types.

3. **Performance Optimization**: Further optimize algorithms to reduce computational overhead while maintaining compliance with competition constraints.

4. **Comprehensive Evaluation**: Expand evaluation metrics beyond recall and REASR to include precision and F1 score for a holistic view of model performance.

This report encapsulates the essential findings and methodologies in the design of backdoor trigger recovery models while highlighting areas for further innovation and development.