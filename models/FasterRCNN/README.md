# Faster R-CNN model code

This project implements Faster R-CNN from scratch using PyTorch components, applied to bone fracture detection on the FracAtlas dataset.
The goal is to deeply understand each step of the pipeline:
	•	Custom dataset loader following COCO-style annotations
	•	Manual backbone construction using ResNet-50 (C5 features)
	•	Manual anchor generation
	•	Manual ROI pooling
	•	Custom training loop + per-step & per-epoch logging
	•	Custom evaluator for mAP50, mAP50–95, MAR, and mean IoU
	•	Visualization of predictions for qualitative assessment

This implementation mirrors the original Faster R-CNN architecture but avoids the high-level PyTorch wrapper to expose all internal logic.

A second notebook (provided separately) contains the standard torchvision FasterRCNN model for performance comparison.
