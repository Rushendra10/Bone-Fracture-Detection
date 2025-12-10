# Bone-Fracture-Detection (G12)

Welcome to our GitHub repo for our CSCI 5561 Final Project: Bone Fracture Detection in Musculoskeletal Radiographs. 

# Introduction

Bone fractures are common injuries generally diagnosed through X-rays. Deep learning models for fracture detection can aid clinicians in timely and accurate diagnoses of fractures. However, these models typically focus on specific regions of the body, limiting their generalizability. To address this, we trained Faster R-CNN,  RetinaNet, and YOLO12n (object detection models) on the FracAtlas dataset, which consists of X-rays of several regions of the body. Faster R-CNN and RetinaNet achieved average IoUs of 0.28 and 0.53, and mAP@0.5 scores of 0.36 and 0.67, respectively. YOLOv12n achieved an mAP@0.5 score of 0.62. We found that using an ImageNet pretrained RetinaNet model achieved superior performance. 

