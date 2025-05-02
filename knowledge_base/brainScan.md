# BrainScan.AI – MRI-based Brain Tumor Detector

## Overview
BrainScan.AI is a deep learning-powered diagnostic tool that detects brain tumors from MRI scans with high accuracy. It supports early detection through automated image classification using convolutional neural networks (CNNs).

## Problem Statement
Timely and accurate diagnosis of brain tumors is critical for treatment planning. Manual MRI analysis can be time-consuming and error-prone. An AI-driven solution helps radiologists prioritize cases and reduce oversight.

## Solution
The system classifies MRI brain scans into tumor-present or tumor-absent categories using a CNN architecture fine-tuned on public medical imaging datasets. It outputs predictions with confidence scores and visual model interpretability.

## Technologies Used
- Python (TensorFlow, Keras, OpenCV)
- CNN model architecture (ResNet-based or custom)
- Streamlit for live demo interface
- Grad-CAM for visual explanation

## Key Features
- Upload MRI image and get instant diagnosis
- Grad-CAM visualization of model attention
- Batch and real-time classification support
- Lightweight, portable Streamlit app

## Results / Outcomes
- Validation accuracy: 94–96%
- High precision/recall across tumor classes
- Visual heatmaps improve model trustworthiness

## Future Improvements
- Expand to multi-class tumor classification
- Integrate with PACS systems in hospitals

## Demo
[Streamlit App 🔗](#)  
Or launch via `brainscan.html` on your portfolio
