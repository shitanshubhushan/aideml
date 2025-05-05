# Technical Report: Temporal Action Localization for ECCV 2024 Workshop

## Introduction
This report summarizes the empirical findings and technical decisions made during the development of a temporal action localization method for the Second Perception Test Challenge at the ECCV 2024 Workshop. The challenge requires accurate localization and classification of actions within untrimmed videos, utilizing a multimodal training dataset.

## Preprocessing
- **Data Setup**: Utilized a multimodal training dataset consisting of 1608 videos featuring both action and sound annotations. Validation (401 videos) and test sets (5359 videos) were employed for tuning and final evaluation respectively.
- **Feature Extraction**: Pre-extracted video and audio features were utilized, abiding by the constraint to avoid external datasets or annotations.

## Modeling Methods
- **Baseline Review**: The initial method employed was the ActionFormer, a single-stage transformer-based approach which processes features through local self-attention layers for temporal resolution.
  
### Method Extensions
1. **Model Architecture**: Enhanced the transformer encoder by:
   - Integrating deeper attention layers to capture complex dynamics.
   - Implementing a pyramidal representation to better capture actions across varying scales.

2. **Boundary Refinement**: Introduced a novel boundary regression technique to improve detection accuracy of start and end timestamps.

3. **Multimodal Fusion**: Investigated improved fusion techniques between video and audio modalities for richer feature representation.

4. **Evaluation Strategy**: Incorporated Soft-NMS to refine final action segment outputs by reducing overlaps.

## Results Discussion
- **Performance Metric**: The primary assessment criterion was the Mean Average Precision (mAP) computed over various Intersection over Union (IoU) thresholds.
- **Comparative Analysis**: Initial tests indicated significant improvements over the baseline performance. Further tuning of hyperparameters led to enhanced segment detection accuracy.

### Findings
- The modified architecture showed improved mAP scores, validating the effectiveness of the proposed boundary regression adjustments and attention enhancements.

## Future Work
- **Attention Mechanisms**: Explore advanced attention mechanisms to further improve contextual understanding of actions within videos.
- **Regularization Techniques**: Investigate novel regularization methods to mitigate overfitting during training.
- **Comprehensive Testing**: Extend testing to explore varying video qualities and content complexities to ensure robustness across diverse scenarios.
- **Community Contributions**: Publish findings and share base methods with the community to facilitate collaborative advancements in action localization technologies. 

This report captures the pivotal stages and technical nuances encountered during the exploration of temporal action localization, setting a foundation for future research and development in this domain.