# Anomaly Detection in Manufacturing using Autoencoder with Pre-trained ResNet-50

## Overview

This project focuses on utilizing a pre-trained ResNet-50 model as part of an autoencoder architecture for anomaly detection in manufacturing. The autoencoder is trained on normal manufacturing data with augmented imgaes and can identify anomalies by reconstructing input images and calculating reconstruction errors.


## Introduction

Anomaly detection is a crucial aspect of quality control in manufacturing. This project employs an autoencoder with a pre-trained ResNet-50 model for feature extraction and reconstruction. By comparing the reconstructed images with the original ones, anomalies are detected based on the reconstruction errors.


## Description

### Methodology

- **Autoencoder Architecture**: Utilizes a pre-trained ResNet-50 model within an autoencoder architecture for feature extraction and image reconstruction.
- **Anomaly Detection**: Calculates the reconstruction error between original and reconstructed images to identify anomalies.
- **Bounding Box Visualization**: Locates anomaly regions by drawing rectangles around areas with high reconstruction errors.
- **Image Segmentation**: Utilizes contour detection to segment and isolate anomalies in the images.

### Dataset Used

The [MVTec dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad), known for its comprehensive collection of real-world textures and objects, has been used for training and evaluation. This dataset contains images of various materials and objects captured under different conditions, enabling robust anomaly detection models.


### Implementation

The code employs Python with TensorFlow and OpenCV libraries for image processing and computer vision tasks. The process involves:

1. **Loading Pre-trained ResNet-50**: Importing the ResNet-50 model with pre-trained ImageNet weights.
2. **Autoencoder Setup**: Constructing an autoencoder architecture using the ResNet-50 layers.
3. **Anomaly Detection**: Calculating reconstruction errors and identifying anomaly areas.
4. **Visualization**: Displaying original images, reconstructed images, bounding boxes around anomalies, and segmented anomaly areas.

## Usage

### Requirements

- Python 3.x
- TensorFlow
- OpenCV
- Matplotlib
- NumPy

### Instructions

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the anomaly detection script.

## Results

The system successfully identifies anomalies in manufacturing images from the MVTec dataset, providing visualizations of the detected defects.


## Future Improvements

- Experimentation with different pre-trained models for improved feature extraction.
- Fine-tuning the thresholding and anomaly detection parameters for better accuracy.
- Enhancing the segmentation algorithm for precise anomaly isolation.

## Contribution

Contributions and suggestions are welcome! Feel free to open issues or pull requests.

## License

This project is licensed under [MIT License](LICENSE).

