# Technical Report: Temporal Action Localization for ECCV 2024 Workshop

## Introduction
This report summarizes the approach taken for the Second Perception Test Challenge on Temporal Action Localization, focusing on methods to accurately localize and classify actions in untrimmed videos. Our goal was to develop a system that improves upon the baseline ActionFormer approach using the provided multimodal training data.

## Preprocessing
- **Data Preparation**: Utilized the multimodal training dataset comprising 1,608 videos with both video and audio annotations.
- **Feature Extraction**: Employed pretrained audio and video features to facilitate input for our modeling methods.
- **Normalization**: Applied feature normalization techniques to standardize the input data, enhancing model training stability.

## Modeling Methods
- **Baseline Method**: ActionFormer
  - Implemented a single-stage transformer-based model using local self-attention to process extracted video and audio features.
  - Multi-scale representation was constructed to capture actions at varying temporal resolutions.

### Proposed Modifications
- **Advanced Boundary Refinement**: Enhanced the boundary regression component to improve action segment accuracy.
- **Attention Mechanism Innovations**: Experimented with hierarchical attention structures to better capture contextual action dependencies.
- **Multimodal Fusion Techniques**: Developed an integrated approach for combining audio and video features more effectively to improve classification performance.
- **Regularization Techniques**: Implemented novel loss functions to mitigate overfitting while maintaining robust training dynamics.

## Results Discussion
- **Evaluation Metrics**: Performance was primarily assessed using Mean Average Precision (mAP) across varying Intersection over Union (IoU) thresholds (0.1 to 0.5).
- **Preliminary Findings**: Initial models demonstrated an mAP improvement over the baseline. The advanced boundary refinement and multimodal fusion methods contributed significantly to detecting more accurate action segments.

### Challenges Faced
- **Overlapping Detections**: Addressed using Soft-NMS and other post-processing techniques to minimize false positives.
- **Training Convergence**: Encountered issues related to model convergence which were alleviated through careful tuning of learning rates and batch sizes.

## Future Work
- **Further Hyperparameter Optimization**: More extensive tuning of model hyperparameters could yield additional performance improvements.
- **Exploration of Additional Architectures**: Investigate the potential of alternative transformer models or hybrid architectures combining CNNs and LSTMs.
- **Integration of Feedback Mechanisms**: Implement mechanisms to iteratively improve segment predictions based on post-inference feedback from validation data.

---

This report encapsulates the empirical findings and technical decisions made during the challenge preparation. Continuous evaluation against the baseline and incorporation of innovative strategies will drive our advancements in temporal action localization.