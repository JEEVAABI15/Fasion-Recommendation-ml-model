
import streamlit as st
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from scipy.spatial.distance import cosine

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

# Load the pre-extracted features from the pickle file
with open('all_features.pkl', 'rb') as f:
    all_features = pickle.load(f)
    
with open('all_image_names.pkl', 'rb') as f:
    all_image_names = pickle.load(f)

# Preprocessing function for images
def preprocess_image(img):
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

# Feature extraction function
def extract_features(model, preprocessed_img):
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

# Define the recommendation function
def recommend_fashion_items_cnn(input_image, all_features, all_image_names, model, top_n=5):
    # Pre-process the input image and extract features
    preprocessed_img = preprocess_image(input_image)
    input_features = extract_features(model, preprocessed_img)

    # Calculate similarities and find the top N similar images
    input_features_reshaped = input_features.reshape(1, -1)

    # Flatten for cosine similarity
    # Calculate similarities and find the top N similar images
    similarities = [1 - cosine(input_features_reshaped, other_feature.reshape(1, -1)) for other_feature in all_features]
    
    similar_indices = np.argsort(similarities)[-top_n:]

    # Filter out the input image index from similar_indices
    similar_indices = [idx for idx in similar_indices if idx != all_image_names.index(input_image)]

    # Display the input image and recommendations
    st.image(input_image, caption="Input Image", use_column_width=True)
    st.write("Recommendations:")
    for i, idx in enumerate(similar_indices[:top_n], start=1):
        recommendation_image = Image.open(all_image_names[idx])
        st.image(recommendation_image, caption=f"Recommendation {i}", use_column_width=True)

# Main Streamlit app
def main():
    st.title("Fashion Recommendation System")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Process uploaded image
        input_image = Image.open(uploaded_file)

        # Convert uploaded image to a temporary path for processing
        temp_filename = f"temp_image.{uploaded_file.name.split('.')[-1]}"  # Create a temporary filename
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Use the temporary path for feature extraction
        input_image_path = temp_filename  # Use the temporary filename as path
        preprocessed_img = preprocess_image(input_image)
        input_features = extract_features(model, preprocessed_img)

        # Display recommendations
        recommend_fashion_items_cnn(input_image, all_features, all_image_names, model, top_n=4)

        # # Clean up the temporary file after processing
        # import os
        # os.remove(temp_filename)  # Remove the temporary file




if __name__ == "__main__":
    main()