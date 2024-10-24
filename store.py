import streamlit as st
from zipfile import ZipFile
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from scipy.spatial.distance import cosine
import glob

# Function to preprocess image
def preprocess_image(img):
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

# Function to extract features
def extract_features(model, preprocessed_img):
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

# Function to recommend fashion items
def recommend_fashion_items_cnn(input_image, all_features, all_image_names, model, top_n=5):
    # Pre-process the input image and extract features
    preprocessed_img = preprocess_image(input_image)
    input_features = extract_features(model, preprocessed_img)

    # Calculate similarities and find the top N similar images
    similarities = [1 - cosine(input_features, other_feature) for other_feature in all_features]
    similar_indices = np.argsort(similarities)[-top_n:]

    # Filter out the input image index from similar_indices
    similar_indices = [idx for idx in similar_indices if idx != all_image_names.index(input_image.name)]

    # Display the input image and recommendations
    st.image(input_image, caption="Input Image", use_column_width=True)
    for i, idx in enumerate(similar_indices[:top_n], start=1):
        st.image(Image.open(all_image_names[idx]), caption=f"Recommendation {i}", use_column_width=True)

# Main function
def main():
    # Streamlit file uploader for image upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        # Load pre-trained VGG16 model
        base_model = VGG16(weights='imagenet', include_top=False)
        model = Model(inputs=base_model.input, outputs=base_model.output)

        # Load and preprocess images
        image_directory = 'women_fashion/women fashion'
        image_paths_list = [Image.open(file) for file in glob.glob(os.path.join(image_directory, '*.*'))
                            if file.endswith(('.jpg', '.png', '.jpeg', 'webp'))]

        # Extract features from images
        all_features = [extract_features(model, preprocess_image(img)) for img in image_paths_list]
        all_image_names = [os.path.basename(file) for file in image_paths_list]

        # Perform fashion recommendation
        recommend_fashion_items_cnn(uploaded_file, all_features, all_image_names, model)

if __name__ == "__main__":
    main()
