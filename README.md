
# Deep Learning Flower Classifier with TensorFlow

This project demonstrates the use of deep learning techniques for classifying flowers based on their images using a Convolutional Neural Network (CNN) model built with TensorFlow.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Details](#model-details)
6. [Dependencies](#dependencies)

---

## Project Overview

This repository contains a deep learning model designed to classify flower images into predefined categories. The model is trained using **TensorFlow** and **Keras**, with a dataset containing labeled images of flowers such as **daisy**, **dandelion**, **rose**, **sunflower**, and **tulip**. The model can be deployed for real-time flower classification applications.

---

## Features

- **Image Classification**: Classifies images of flowers into categories such as **daisy**, **dandelion**, **rose**, **sunflower**, and **tulip**.
- **Convolutional Neural Network (CNN)**: Utilizes CNN for effective feature extraction and classification.
- **TensorFlow and Keras**: Built with TensorFlow and Keras, providing easy integration and scalability.
- **Model Storage**: The trained model is saved as a `.h5` file, which can be used for inference in real-time applications.

---

## Installation

### Clone the Repository

Clone the repository to your local machine:
```bash
git clone https://github.com/thivsiv/Deep-Learning-Flower-Classifier-TensorFlow.git
cd Deep-Learning-Flower-Classifier-TensorFlow
```

### Set Up Virtual Environment

It is recommended to set up a virtual environment for this project to manage dependencies.

#### Create a Virtual Environment
To create a virtual environment, run the following command in your terminal:
```bash
python -m venv venv
```

#### Activate the Virtual Environment

- **For Windows**:
  ```bash
  venv\Scripts\activate
  ```

- **For macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

Once activated, your terminal prompt should change to indicate that you're working inside the virtual environment.

### Install Dependencies

Install the necessary Python libraries using pip:
```bash
pip install -r requirements.txt
```

If you don't have the `requirements.txt` file, you can manually install the dependencies as follows:
```bash
pip install tensorflow keras numpy matplotlib scikit-learn opencv-python pillow
```

### Verify Installation

To verify that TensorFlow and other dependencies are installed correctly, you can run the following Python command to check TensorFlow's version:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

This should return the installed version of TensorFlow.

---

## Usage

### Train the Model

To train the model, run the following command:
```bash
python train_model.py
```

This will initiate the training process, where the model will learn from the dataset and be saved as `flower_classifier_model.h5`.

### Model Inference

Once the model is trained, you can use it to make predictions on new flower images. Run:
```bash
python classify_flower.py --image path/to/flower_image.jpg
```

The model will classify the image and return the flower category.

---

## Model Details

The model is based on a **Convolutional Neural Network (CNN)** architecture, effective for image classification tasks. The steps involved in this project include:

### Data Preprocessing
- Images are resized, normalized, and augmented to improve the model's ability to generalize.

### Model Architecture
- A CNN with multiple convolutional layers, max-pooling layers, and fully connected layers.

### Training
- The model is trained using a **categorical cross-entropy** loss function with the **Adam optimizer**.

### Model Evaluation
- After training, the model is evaluated on a test set for accuracy.

The trained model is saved as `flower_classifier_model.h5`.

---

## Dependencies

This project requires the following Python libraries:

- **TensorFlow** >= 2.0
- **Keras**
- **NumPy**
- **Matplotlib**
- **scikit-learn**
- **OpenCV**
- **Pillow**

To install the dependencies, run:
```bash
pip install -r requirements.txt
```

Alternatively, you can manually install these dependencies using pip:
```bash
pip install tensorflow keras numpy matplotlib scikit-learn opencv-python pillow
```
