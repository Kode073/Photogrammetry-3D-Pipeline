#  3D Photogrammetry Pipeline from Scratch

##  Overview
This repository contains an end-to-end automated 3D photogrammetry pipeline developed as part of a major engineering project at IMT Atlantique. The goal of this project was to understand and implement the core mathematical concepts of photogrammetry from scratch, rather than relying on black-box software. 
The program are in the program folder. The rest are images or result for the presentation in class. 

The pipeline takes a series of 2D images as input and reconstructs a 3D point cloud/model, leveraging custom mathematical implementations and Deep Learning for feature matching.

##  Key Features
* **Custom Mathematical Implementation:** Coded the fundamental photogrammetry equations and concepts from scratch in Python.
* **AI-Powered Feature Matching:** Integrated a custom AI model using **PyTorch** for robust keypoint detection and matching across image pairs.
* **Semi-Automated Pipeline:** The system successfully processes image series with minor human assistance required for final camera calibration.

##  Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch
* **Computer Vision / 3D:** OpenCV
* **Version Control:** Git


## Pipeline Architecture
1. **Data Ingestion:** Loading a dataset of overlapping 2D images.
2. **Feature Detection & Matching (PyTorch):** Identifying common points across multiple views.
3. **Camera Pose Estimation & Calibration:** Calculating the exact position and angle of the camera for each shot.
4. **Triangulation:** Projecting the matched 2D points into a 3D space.

