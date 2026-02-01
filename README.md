# College Project: Object Detection and Tracking using YOLOv5

## Project Description

This GitHub repository contains the complete implementation of my **college academic project** on **object detection and multi-object tracking using deep learning techniques**.

The project uses the **YOLOv5 framework** for object detection and demonstrates how detection models can be trained on benchmark datasets and then applied for tracking objects across video frames.

All coding work carried out for this project is included in this repository, as per the submission instructions.

---

## Aim of the Project

The main aims of this project are:
- To understand and implement the YOLOv5 object detection framework
- To train a custom object detection model on the MOT17 dataset
- To apply object tracking techniques on video data using detection outputs
- To gain hands-on experience with real-world computer vision pipelines

---

## Repository Contents

.
├── YOLOv5_model_MOT17_training.ipynb
├── object_tracking.ipynb
├── README.md


---

## File-wise Explanation

### 1. YOLOv5_model_MOT17_training.ipynb

This notebook contains the **training and experimentation phase** of the project.

**Contents:**
- Setup of the YOLOv5 framework
- Preparation and preprocessing of the MOT17 dataset
- Conversion of annotations into YOLO format
- Creation of dataset configuration file
- Training YOLOv5 models for pedestrian detection
- Training multiple YOLOv5 variants
- Saving trained weights and training logs

**Purpose:**  
To train a deep learning–based object detection model on a real-world benchmark dataset.

---

### 2. object_tracking.ipynb

This notebook focuses on the **application phase** of the project.

**Contents:**
- Loading a pretrained YOLOv5 detection model
- Performing object detection on video frames
- Using ByteTrack for multi-object tracking
- Assigning consistent IDs to detected objects
- Applying CLIP for zero-shot object classification
- Visualizing detection and tracking results

**Purpose:**  
To demonstrate how object detection models can be extended to object tracking in video sequences.

---

## Methodology

1. Preparation of the MOT17 dataset
2. Training YOLOv5 object detection models
3. Performing object detection on video frames
4. Applying tracking algorithms to maintain object identities
5. Visualization and qualitative analysis of results

---

## Results

- YOLOv5 successfully detects pedestrians in complex scenes
- Trained models produce accurate bounding boxes
- ByteTrack provides consistent object IDs across frames
- The detection and tracking pipeline performs effectively on video data

---

## Conclusion

This project successfully demonstrates an end-to-end pipeline for **object detection and multi-object tracking** using YOLOv5.  
The work provided practical experience with dataset preprocessing, deep learning model training, and video-based computer vision analysis.

---

## Tools and Technologies Used

- Python  
- Jupyter Notebook  
- PyTorch  
- YOLOv5  
- OpenCV  
- ByteTrack  
- CLIP  

---

## Academic Declaration

This repository was created as part of a **college academic project submission**.  
All notebooks and code included here were developed and used solely for academic evaluation purposes.
