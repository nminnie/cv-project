# Computer Vision Image Segmentation Project
A comprehensive exploration of image segmentation techniques using traditional and modern deep learning approaches on the Oxford-IIIT Pet Dataset.
## Project Overview
This project implements and compares multiple image segmentation architectures to segment pet images into different classes. The goal is to explore how different pre-training strategies and network architectures affect segmentation performance while keeping models as compact as possible.

### Multiple Architectures
UNet-based end-to-end segmentation - Custom UNet implementation with encoder-decoder structure and skip connections
Autoencoder pre-training - Two-stage training with unsupervised feature learning followed by segmentation fine-tuning
CLIP features for segmentation - Leveraging pre-trained CLIP image encoder for powerful feature extraction
Prompt-based segmentation - Interactive segmentation using point prompts (similar to SAM)

### Robustness Testing
Systematic evaluation against 8 different perturbations:

Gaussian pixel noise
Gaussian blurring
Contrast increase/decrease
Brightness increase/decrease
Random occlusion
Salt and pepper noise

