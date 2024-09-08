import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image

# # Data base

# Operation Github
@st.cache_data
def load_saved_model():
    
    # Models
    modelEdnaModa = load_model("C:/Users/arian/OneDrive/Documentos/GitHub/Artificial-Intelligence-Resources-New/Download_Files/modelsUse/clothingModelAINew.keras") # Download_Files/modelsUse/clothingModelAI.keras

    # Return    
    return modelEdnaModa

# Function to preprocess the image
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 28x28
    image = image.resize((28, 28))
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    # Add dimensions to match the model input
    image_array = image_array.reshape((1, 28, 28, 1))
    return image_array

# Main Streamlit function
def main():

    st.set_page_config(page_title="AI Clothing Classifier", page_icon="👕")

    st.title('AI Clothing Classifier')

    # Widget to upload image
    uploaded_file = st.file_uploader("Choose a clothing image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the image
        image = Image.open(uploaded_file)
        
        # Display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Charge model
        model = load_saved_model()

        # Make the prediction
        prediction = model.predict(processed_image)
        class_index = np.argmax(prediction)

        # Define class names
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        # Display the prediction
        st.write(f"The item in the image appears to be: {class_names[class_index]}")
        st.write(f"Confidence: {prediction[0][class_index]*100:.2f}%")

if __name__ == '__main__':
    main()