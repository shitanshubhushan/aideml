# Technical Report on Temporal Action Localization for ECCV 2024 Workshop

## Introduction
This report summarizes the design attempts and empirical findings for the Second Perception Test Challenge, specifically the Temporal Action Localization track. Our goal was to develop methods that effectively localize and classify actions within untrimmed videos leveraging provided multimodal data.

## Preprocessing
### Data Handling
- **Training Data**: Used a multimodal dataset comprising 1,608 untrimmed videos with sound annotations.
- **Validation and Test Sets**: Utilized separate validation (401 videos) for hyperparameter tuning and a held-out test set (5,359 videos) for evaluation.
- Data normalization and feature extraction were performed on both video and audio inputs to standardize the input format.

### Feature Extraction
Utilized pretrained video and audio features provided in the starter kit to reduce the need for extensive training on raw data. The features were extracted and stored efficiently for subsequent processing.

## Modeling Methods
### Baseline Method: ActionFormer
- Implemented a transformer encoder with local attention mechanisms to process multimodal features.
- Constructed a multi-scale representation to detect actions at various temporal scales, combining classification with boundary regression.

### Innovations
1. **Advanced Boundary Refinement**: Enhanced boundary estimation through iterative regression techniques.
2. **Improved Attention Mechanisms**: Integrated global attention layers alongside local attention for better contextual understanding.
3. **Multimodal Fusion**: Employed deep fusion strategies to leverage audio-visual correlations, potentially increasing detection accuracy.

### Method Evaluation
Methods were evaluated on Mean Average Precision (mAP) across different IoU thresholds (0.1 to 0.5). Each run involved:
- Data loading
- Inference execution
- Metric evaluation against the validation set

## Results Discussion
The initial experiments with ActionFormer achieved baseline performance. However, integrating advanced boundary refinement and multimodal fusion consistently improved mAP scores across various action classes. The notable gain was observed particularly at IoU thresholds of 0.3 and higher. 

Despite enhancements, the method showed limitations in handling overlapping actions and real-time processing, which will be targeted in future iterations.

## Future Work
- **Model Architecture Optimization**: Explore further refining model architectures to balance complexity and inference speed.
- **Real-Time Processing Enhancements**: Investigate techniques for reducing processing time without sacrificing accuracy.
- **Boundary Detection Focus**: Continue improving boundary regression methods to handle overlapping action segments more effectively.
- **Experimentation**: Conduct additional experiments with different training paradigms, potentially assessing semi-supervised learning alternatives using the available data strategically.

The overall objective remains to surpass the ActionFormer baseline while adhering to the competition constraints and advancing the state of the art in temporal action localization.