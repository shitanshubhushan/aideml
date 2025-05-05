# Technical Report on Temporal Action Localization for ECCV 2024 Workshop

## Introduction
This report summarizes the design attempts and outcomes associated with developing a temporal action localization model for the Second Perception Test Challenge. The goal is to localize and classify actions in untrimmed videos using a transformer-based approach, building upon the ActionFormer baseline.

## Preprocessing
### Data Preparation
- **Training Dataset**: Utilized a multimodal training set containing 1,608 videos with action and sound annotations.
- **Validation and Test Sets**: Employed 401 videos for hyperparameter tuning and a separate test set of 5,359 videos for final evaluation.

### Feature Extraction
- Extracted video and audio features using pretrained models to conform to the competition constraints and enhance model performance.

## Modeling Methods
### Baseline Model
- Implemented ActionFormer, a transformer-based model with:
  - **Local Attention Mechanism**: Captures actions through a multi-scale representation.
  - **Action Classification & Boundary Regression**: Classifies each time frame and regresses start and end boundaries of actions.
  - **Decoding Techniques**: Utilized Soft-NMS to refine action segment predictions.

### Modified Approaches
1. **Enhanced Attention Mechanisms**: 
   - Investigated self-attention improvements to better capture long-term dependencies in video.
  
2. **Boundary Refinement Techniques**: 
   - Explored novel boundary regression strategies for higher segment accuracy.

3. **Multimodal Fusion Strategies**: 
   - Integrated audio features more effectively with video features to improve action classification.

### Implementation Steps
- Created `MyNewMethod.py` and modified core functions to implement innovations.
- Registered this method in `__init__.py` for compatibility within the codebase.

## Results Discussion
- **Evaluation Metric**: Mean Average Precision (mAP) was computed across different IoU thresholds.
- Achievements:
  - The modified approach demonstrated improved mAP over the ActionFormer baseline in validation tests.
  - Notable enhancements in precision, particularly in complex action classes.
  
- Challenges Encountered:
  - Struggled with overlaps in detected action segments; addressed with refined decoding methods.
  - Balancing computational efficiency while maintaining model performance.

## Future Work
- Further explore advanced neural architectures, such as incorporating graph neural networks for better relationships between actions.
- Investigate unsupervised techniques for better boundary refinement using existing unlabeled video data.
- Conduct extensive ablation studies to identify effective components of the model architecture.

This report lays a foundation for ongoing developments in temporal action localization, aiming to push the boundaries of current methodologies within the ECCV framework.