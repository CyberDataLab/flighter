## Overview of Models

In our manuscript, we utilized two primary models to perform comprehensive image classification tasks: a lightweight Convolutional Neural Network (CNN) and the VGG16 architecture. This document outlines the structure, advantages, and purpose of each model, as well as the training methodology used. We further discuss the extensibility of our framework to accommodate future research involving additional models and datasets.

---

## 1. Lightweight Convolutional Neural Network (CNN)

### Model Description
The lightweight Convolutional Neural Network (CNN) employed in our study features a series of convolutional layers arranged into blocks, designed to efficiently extract and process image features. This model architecture is purposefully lightweight to facilitate real-time processing in resource-constrained environments, such as onboard aircraft.

### Architectural Details

#### Layer Breakdown
1. **First Convolutional Block**
   - **Kernel Size**: 5x5
   - **Input Channels**: The number of input channels specific to the dataset used
   - **Output Channels**: 16 feature maps
   - **Activation Function**: ReLU (Rectified Linear Unit)
   - **Pooling**: Max-pooling applied to reduce the spatial dimensions

2. **Second Convolutional Block**
   - **Kernel Size**: 5x5
   - **Input Channels**: 16
   - **Output Channels**: 32 feature maps
   - **Activation Function**: ReLU
   - **Pooling**: Max-pooling applied to further down-sample the feature maps

3. **Third Convolutional Block**
   - **Kernel Size**: 6x6
   - **Input Channels**: 32
   - **Output Channels**: 64 feature maps
   - **Activation Function**: ReLU
   - **Pooling**: Max-pooling applied to continue reducing dimensionality

4. **Fourth Convolutional Block**
   - **Kernel Size**: 5x5
   - **Input Channels**: 64
   - **Output Channels**: 128 feature maps
   - **Activation Function**: ReLU
   - **Pooling**: Not applied, as we preserve the feature map size for further processing

5. **Dropout Layer**
   - **Dropout Rate**: Specified by `self.dropout_rate`
   - **Purpose**: Reduces overfitting by randomly setting a fraction of the input units to zero during training

6. **Final Convolutional Block**
   - **Kernel Size**: 3x3
   - **Input Channels**: 128
   - **Output Channels**: The number of classes for the classification task
   - **Activation Function**: Not explicitly used, as this block outputs the class logits

7. **Flatten Layer**
   - **Purpose**: Converts the multi-dimensional feature maps into a 1D tensor, suitable for classification

### Purpose and Use Case
The lightweight CNN is optimized for computational efficiency while maintaining high performance in image classification tasks. It is particularly well-suited for deployment in environments where hardware resources are limited, such as onboard aircraft. The model’s architecture effectively balances feature extraction and computational demands, making it ideal for real-time applications.

### Implementation Notes
- **Custom Convolutional Blocks**: The convolutional blocks (`_blocks.Conv2DBlock`) are custom-defined to streamline the integration of convolution, activation, and pooling operations.
- **Flexibility**: The model is highly modular, allowing easy adjustments to layers, filter sizes, and pooling strategies to suit different datasets or computational constraints.

---

## 2. VGG16 Architecture

### Model Description
The VGG16 model serves as a deeper and more complex benchmark in our study. With 16 layers, VGG16 is a well-established architecture known for its ability to capture intricate features from images, leveraging a series of convolutional layers followed by fully connected layers.

### Architectural Details
- **Layer Composition**: The VGG16 architecture consists of 13 convolutional layers and 3 fully connected layers.
- **Convolutional Layers**: Employ small 3x3 filters throughout the network, preserving the spatial information of the images.
- **Pooling Strategy**: Max-pooling layers are used to progressively reduce the spatial dimensions of the feature maps.
- **Fully Connected Layers**: Three fully connected layers aggregate features and perform the final classification.

### Purpose and Use Case
VGG16 was selected as a benchmark to compare against the lightweight CNN. Its depth and extensive feature extraction capabilities make it well-suited for complex image classification tasks, such as Synthetic Aperture Radar (SAR) image analysis. Despite its higher computational cost, VGG16 provides a valuable performance reference.

---

## Extensibility and Future Work

Our framework is designed to be extensible, allowing the integration of additional models and datasets to facilitate broader research endeavors. Although our study focused on the lightweight CNN and VGG16, future investigations could incorporate models such as ResNet, DenseNet, or custom architectures tailored to specific data characteristics. The modular nature of our implementation ensures flexibility for further advancements and comparative analysis.

---

## Training and Evaluation Insights

### Training Methodology
Both models were trained using Stochastic Gradient Descent (SGD) to achieve optimal convergence rates. This optimizer was selected for its proven track record in efficiently minimizing loss functions, especially in deep learning applications. The training process included:
- **Hyperparameter Adjustments**: Fine-tuning learning rates, batch sizes, and momentum to enhance model performance.
- **Model Evaluation**: Each model's performance was assessed using established metrics, with F1 scores serving as a key indicator of classification efficacy.

### Performance Metrics
- **Baseline F1 Score**: The manuscript includes a comparative analysis of the baseline F1 score achieved by our models across different datasets and under each specified 𝛼 distribution, as depicted in Fig. 4.
