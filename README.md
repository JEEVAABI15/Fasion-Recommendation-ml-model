# Fashion Recommendation System

This project is a **Fashion Recommendation System** that uses a pre-trained deep learning model (VGG16) to suggest similar fashion items based on an input image. The system processes an uploaded image, extracts features using Convolutional Neural Networks (CNN), and compares them with pre-extracted features of other fashion images to generate recommendations.

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Feature Extraction](#feature-extraction)
- [Cosine Similarity and Recommendations](#cosine-similarity-and-recommendations)
- [Streamlit Application](#streamlit-application)
- [Installation](#installation)
- [How to Run](#how-to-run)

## Overview

This project provides a content-based image recommendation system for women's fashion. It takes an image of a fashion item as input and recommends the top 4 similar fashion items based on feature extraction using a pre-trained VGG16 model. The frontend is built using **Streamlit**, allowing users to upload images and view the recommended fashion items.

## Technologies Used

- **Python 3.10+**
- **TensorFlow 2.x**
- **Keras**
- **Streamlit**
- **Pandas**
- **NumPy**
- **SciPy (Cosine Similarity)**
- **Pillow (PIL)**
- **Matplotlib**

## Dataset

The dataset used consists of women's fashion images stored in a ZIP file. Each image is extracted and preprocessed before feature extraction using the VGG16 model.

## Feature Extraction

1. **VGG16 Model**: A pre-trained VGG16 model, available in TensorFlow, is used to extract features from the images. The final layer of the network is removed, and the output of the convolutional layers is used as the feature vector.
2. **Normalization**: After extracting the features, they are flattened and normalized to ensure uniform scaling for cosine similarity calculations.

## Cosine Similarity and Recommendations

The system uses **cosine similarity** to compare the input image's feature vector with the pre-extracted feature vectors from the dataset. The `top N` images with the highest similarity scores are recommended to the user.

## Streamlit Application

The frontend of the system is developed using **Streamlit**, which allows users to upload an image, process it, and view the recommendations. The core steps include:
- Upload an image via the file uploader.
- Extract features from the uploaded image.
- Calculate cosine similarity with pre-extracted features.
- Display the input image and the top 4 recommended fashion items.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/fashion-recommendation-system.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset and place it in the project directory.

## How to Run

1. **Run the Streamlit application**:

    ```bash
    streamlit run app.py
    ```

2. **Upload an image**: Once the app is running, you can upload an image through the Streamlit interface.
3. **View recommendations**: After uploading the image, the system will display the top 4 similar fashion items based on the uploaded image.

## Future Enhancements

- Support for more fashion categories (e.g., men's fashion, accessories).
- Use of other pre-trained models for feature extraction.
- Addition of a more interactive UI for filtering and refining recommendations.
