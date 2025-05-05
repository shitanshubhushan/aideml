# Technical Report on Temporal Action Localization for ECCV 2024 Workshop

## Introduction
This report summarizes the design attempts and empirical findings in developing a model for the Temporal Action Localization Track of the Second Perception Test Challenge. The objective is to accurately localize and classify actions in untrimmed videos, drawing from a multimodal dataset containing annotated audio and video features.

## Preprocessing
### Data Overview
- **Training Set**: 1,608 videos with action and sound annotations.
- **Validation Set**: 401 videos for hyperparameter tuning.
- **Test Set**: 5,359 videos, reserved for final model evaluation.

### Feature Extraction
Utilized provided pretrained audio and video features as input. Preprocessing involved normalizing and structuring data into a suitable format for model training.

## Modeling Methods
### Baseline Approach: ActionFormer
1. **Architecture**: Transformer encoder with local attention for processing multimodal features.
2. **Representation**: Constructed a multi-scale representation via stacked self-attention layers to capture varying temporal scales.
3. **Classification & Regression**: Each time step is classified as action or background, with distance regressions to action segment boundaries.
4. **Post-processing**: Implemented Soft-NMS to consolidate overlapping action detections.

### Innovations Explored
1. **Boundary Refinement**: Enhanced boundary regression techniques resulted in tighter segment boundaries.
2. **Attention Mechanisms**: Experimented with different self-attention configurations for improved context capture across longer durations.
3. **Multimodal Fusion Techniques**: Various fusion strategies explored to better combine audio and video signals, leading to increased classification accuracy.

## Results Discussion
### Performance Metrics
- **Mean Average Precision (mAP)** evaluated across IoU thresholds, revealing consistent improvements over baseline, particularly with refined models.
- Significant variations in performance metrics were observed on the validation data.
  
### Key Findings
- Implementing advanced boundary-regression techniques positively influenced boundary precision.
- Improved attention mechanisms yielded better context understanding, significantly enhancing detection accuracy in complex scenarios.

## Future Work
- **Continued Innovation**: Aim to explore novel loss functions to further refine model training.
- **Enhanced Fusion Techniques**: Test additional multimodal fusion methods to maximize the combination of audio and video signals.
- **Robust Evaluation**: Perform extended evaluations across diverse unseen datasets to assess model generalization and robustness.

The experiments show promising results toward developing a competitive action localization system, paving the way for further advancements in this field.