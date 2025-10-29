# Image-Classification-Sysytem
Convolutional Neural Networks for Image Recognition

This project focuses on supervised image classification of color images using Convolutional Neural Networks (CNNs). It leverages modern deep learning frameworks and employs data augmentation to accurately identify images across ten predefined categories.

📊 Dataset — CIFAR-10

The CIFAR-10 dataset, a benchmark in computer vision research, was used to train and evaluate the model.

Key Features:

60,000 RGB images of size 32×32 pixels, categorized into 10 distinct classes.

50,000 training and 10,000 testing samples.

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

⚙️ Project Workflow

The implementation covers end-to-end training pipelines using TensorFlow and PyTorch within the notebooks
 directory.

1️⃣ CIFAR-10 with TensorFlow

📘 Notebook: image-classification-tensorflow.ipynb

Process Overview:

Setup: Import required libraries and dependencies.

Data Loading: Fetch the dataset via keras.datasets.cifar10.

Preprocessing & Augmentation:

Normalize pixels to [0, 1].

Apply shear, zoom, and random horizontal flips using ImageDataGenerator.

Model Construction:

Sequential CNN architecture with multiple Conv-ReLU-MaxPool blocks.

Flatten and connect through fully-connected Dense layers.

Final Softmax layer for classification across 10 classes.

Training:

Optimizer: Adam

Loss: Categorical Crossentropy

Batch size: 32, Epochs: 25

Evaluation:

Visualize training/validation accuracy and loss over epochs.

Analyze overfitting and performance metrics.

2️⃣ CIFAR-10 with PyTorch

📘 Notebook: image-classification-pytorch.ipynb

Process Overview:

Initialization: Import torch, torchvision, and relevant modules.

Dataset & Transformations:

Load CIFAR-10 using torchvision.datasets and DataLoader.

Apply transformations: random rotation, flip, color jitter, and normalization.

Model Architecture:

Custom CNN built using torch.nn.Module.

Includes Conv2D + ReLU + MaxPool layers, followed by Dense layers and a Softmax output.

Optimizer: Adam, Loss: CrossEntropyLoss.

Training:

25 Epochs, tracked using training/validation accuracy plots.

Performance Visualization:

Accuracy and loss graphs over epochs to compare with TensorFlow results.

✅ Outcome:
Both TensorFlow and PyTorch implementations effectively classify CIFAR-10 images, demonstrating how CNNs extract and learn spatial patterns from color image datasets.
