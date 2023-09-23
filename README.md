# Real-Time Firearm Detection from CCTV Footage

## Introduction
In the interest of public safety, early detection of potentially violent situations is paramount. One proactive approach is to identify the presence of firearms in surveillance footage. This project aims to detect firearms in real-time from CCTV videos, highlighting them within bounding boxes for immediate attention.

## Problem Statement
The goal is to develop a model that can process CCTV footage and identify firearms frame-by-frame, marking their presence with a bounding box. This early warning system can be instrumental in preventing potential threats and ensuring public safety.

## Dataset
We utilize the "Guns Object Detection" dataset available on [Kaggle](https://www.kaggle.com/datasets/issaisasank/guns-object-detection). This dataset comprises:
- Labeled images of firearms sourced from various online platforms.
- Square reconstructed images retained in RGB/BGR format.
- The `Images` folder containing jpeg photographs.
- The `Labels` folder with corresponding text files detailing object coordinates.

Each text file starts with a list of objects from its corresponding image and concludes with the object's coordinates. This dataset serves as the foundation for our training.

## Algorithm
The YOLOv3 (You Only Look Once version 3) algorithm is our choice for firearm detection. Key highlights of our approach include:
- Training on a dataset with a singular class: firearms.
- Real-time identification of weapon types from CCTV footage post-training.
- Immediate threat alerts to relevant authorities upon weapon detection.
- YOLOv3's unique architecture that combines feature extraction and object localization into a unified block, ensuring rapid inference.
- Unlike traditional methods that scan images with a sliding window, YOLO processes the entire image through a Convolutional Neural Network (CNN) in a single pass, predicting bounding box coordinates and class probabilities simultaneously.
- Our dataset, housed in the `darknet` folder, is uploaded to a drive, segmented into 70% training and 30% testing data, and organized for continuous model training.
- We leverage OpenCV's deep neural network module, which incorporates Darknet's deep learning framework (where YOLO originates). Our code loads the network, processes the YOLOv3 configuration and weights, and reads an input image. Post neural network processing, predictions are refined by removing low-confidence detections and applying non-max suppression to eliminate redundant bounding boxes, ensuring optimal object detection. Weapon detection triggers an alert.

## Execution
Follow these steps to execute the firearm detection:

### Requirements:
- Python 3.8 or newer.
- Ensure all dependencies are installed using: 

$ pip3 install -r requirements.txt

- Avoid altering configuration or weight files; they result from training YOLO v3 on the Kaggle gun dataset.

### Steps:
1. The training code is available in the `Training_YoloV3_firearm_detection.ipynb` Jupyter notebook. Execute this notebook on Google Colab to generate the necessary configurations and weights.
2. `main.py` facilitates firearm detection in three scenarios:
   - Image
   - Video
   - Real-time
3. For code execution, run the `run.sh` file. By default, `main.py` uses inputs from the root directory for gun detection.

## References:
1. [Introduction to YOLO Algorithm for Object Detection](https://www.section.io/engineering-education/introduction-to-yolo-algorithm-for-object-detection/)
2. [Object Detection with YOLOv3 in Keras](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/)
3. [YOLO Object Detection with OpenCV](https://pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv)
