# Trash Classification with Deep Learning

This repository contains a deep learning project aimed at classifying types of trash using the [TrashNet dataset on Hugging Face](https://huggingface.co/datasets/garythung/trashnet). The project includes data preparation, model development, and deployment steps, leveraging GitHub Actions for CI/CD automation and Weights & Biases (W&B) for model tracking.

## Project Overview

The goal of this project is to develop and deploy an image classification model capable of categorizing trash images into distinct classes. This is achieved through a Convolutional Neural Network (CNN) trained on the TrashNet dataset.

### Key Deliverables
1. **Image Classification Model**: A deep learning model trained to classify trash types, published on Hugging Face Hub.
2. **GitHub Repository**: Contains the Jupyter notebook and scripts necessary for development, along with a guide for reproducing the model.
3. **CI/CD with GitHub Actions**: Automated workflows that streamline model development.
4. **Model Tracking and Versioning with W&B**: All training metrics and model versions are tracked using W&B.

## Project Setup and Reproduction

### Prerequisites
- Python 3.7 or higher
- Git
- Jupyter Notebook
- Weights & Biases (W&B) account
- Hugging Face account with API access

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/trash-classification.git
   cd trash-classification
2. Install dependencies:
3. ```bash
   pip install -r requirements.txt
4. Log in to W&B and Hugging Face to enable model tracking and deployment:
   ```bash
   wandb login
   huggingface-cli login
   

## Running the Notebook
1. Open `Trash_Classification.ipynb` in Jupyter Notebook.
2. Follow the notebook steps for data preparation, model training, and evaluation.

### Contents:
- `Trash_Classification.ipynb`: Jupyter notebook with step-by-step instructions on data preparation, exploratory data analysis, model training, evaluation, and insights.
- `requirements.txt`: List of dependencies required to run the project.
- `.github/workflows`: GitHub Actions workflows for automating model training and deployment.
- `scripts`: Directory containing any additional Python scripts used for data processing or model training.

## Approach and Methodology

### 1. Problem Understanding and Justification
A CNN model is selected for its effectiveness in image classification tasks. CNNs can capture spatial hierarchies in images, making them suitable for identifying features in trash images.

### 2. Data Preparation
- Download and prepare the TrashNet dataset.
- Resize images, normalize pixel values, and apply data augmentation.
- Oversample minority classes to address dataset imbalance.

### 3. Exploratory Image Analysis
- Visualize and analyze class distributions and image characteristics.
- Inspect data augmentations to verify diversity in training samples.

### 4. Model Architecture
A CNN based on MobileNetV2 is implemented, with a custom classification head for trash type classification. Layers are frozen in the pre-trained model to retain general image features.

### 5. Model Training & Evaluation
- Model is trained with W&B for real-time metric tracking.
- Evaluation includes accuracy, precision, recall, and F1-score analysis across classes.
- Insights and limitations, such as class imbalances and potential biases, are noted.

## CI/CD with GitHub Actions
This repository uses GitHub Actions to automate the following:
- **Model Training**: A workflow that trains the model on each commit.
- **Model Deployment**: Upon successful training, the model is automatically pushed to the Hugging Face Hub.

## Model Tracking with W&B
All model versions, training metrics, and performance insights are logged with Weights & Biases. This enables easy experimentation tracking, version comparison, and hyperparameter optimization.

## Reproducibility
To reproduce this project:
1. Clone the repository and install dependencies.
2. Follow the steps in the Jupyter notebook.
3. Set up your Hugging Face and W&B integrations to track experiments and publish models.
