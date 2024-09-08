import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image

# Data base

# Operation Github
@st.cache_resource
def load_saved_model():
    # Models
    modelEdnaModa = load_model("Download_Files/modelsUse/clothingModelAINew.keras")
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

    st.write('The accuracy of the AI has been proven to be 89.22%')

    # Advice for users
    st.subheader("Tips for best classification results:")
    st.write("""
    1. Use a clear, well-lit image of a single clothing item.
    2. Ensure the clothing item is centered in the image.
    3. Use a contrasting background.
    4. Avoid complex patterns or multiple items in one image.
    5. The image should be front-facing and not at an angle.
    6. Ideally, the item should be laid flat or worn by a person.
    7. Avoid images with accessories or other objects.
    8. The closer the image is to a 28x28 pixel grayscale format, the better.
    """)

    # Widget to upload image
    uploaded_file = st.file_uploader("Choose a clothing image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the image file
        image = Image.open(uploaded_file)

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

        # Display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)

if __name__ == '__main__':
    main()