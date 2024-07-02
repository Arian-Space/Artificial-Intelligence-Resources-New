from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="AI to predict feelings from text", page_icon="🚢")

st.title("🚢 AI to predict feelings from text 🚢")

st.write(
    "Welcome to this website where you can use my artificial intelligence to predict feelings from text."
)

st.header("Database citation and use")

st.markdown("The Database that we will use is \"dair-ai/emotion\" from huggingface:"
            "(https://huggingface.co/datasets/dair-ai/emotion)"
)

# # Base de datos

# operation Github
@st.cache_data
def load_data():
    
    # Models
    modelLoadedJoblibSad = joblib.load('Download_Files/modelsUse/arianMind/sadModelCritic.pkl')
    modelLoadedJoblibAnger = joblib.load('Download_Files/modelsUse/arianMind/angerModelCritic.pkl')
    modelLoadedJoblibFear = joblib.load('Download_Files/modelsUse/arianMind/fearModelCritic.pkl')
    modelLoadedJoblibJoy = joblib.load('Download_Files/modelsUse/arianMind/joyModelCritic.pkl')
    modelLoadedJoblibLove = joblib.load('Download_Files/modelsUse/arianMind/loveModelCritic.pkl')
    modelLoadedJoblibSurprise = joblib.load('Download_Files/modelsUse/arianMind/surpriseModelCritic.pkl')

    # Math
    tok = joblib.load('Download_Files/mathBoy/tokenizadorSentimental.pkl')
    
    return modelLoadedJoblibSad, modelLoadedJoblibAnger, modelLoadedJoblibFear, modelLoadedJoblibJoy, modelLoadedJoblibLove, modelLoadedJoblibSurprise, tok

modelLoadedJoblibSad, modelLoadedJoblibAnger, modelLoadedJoblibFear, modelLoadedJoblibJoy, modelLoadedJoblibLove, modelLoadedJoblibSurprise, tok = load_data()

st.header("How to use it?")

st.write(
    "We must understand the results from the algorithm:"
)

st.markdown("""
- Sadness ():
- Surprise (😱):
""")

st.write(
    "More details of the model in my repository on Github."
)

# Función para predecir emociones
def preprocess_text(text):
    max_len = 100
    sequence = tok.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    return padded_sequence

# Predecir emoción
def predict_emotion(text):
    processed_text = preprocess_text(text)

    # Predicción modelos
    predictionSad = modelLoadedJoblibSad.predict(processed_text,verbose=0)
    predictionAnger = modelLoadedJoblibAnger.predict(processed_text,verbose=0)
    predictionFear = modelLoadedJoblibFear.predict(processed_text,verbose=0)
    predictionJoy = modelLoadedJoblibJoy.predict(processed_text,verbose=0)
    predictionLove = modelLoadedJoblibLove.predict(processed_text,verbose=0)
    predictionSurprise = modelLoadedJoblibSurprise.predict(processed_text,verbose=0)

    allEmotionPrediction = [predictionSad,predictionAnger,predictionFear,predictionJoy,predictionLove,predictionSurprise]
    
    # Probabilidad de cada uno
    emotionMap = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    bestEmotion = (allEmotionPrediction.sort())[-1]
    guideEmotionMap = []

    for i in allEmotionPrediction:
        if bestEmotion == i:
            guideEmotionMap.append(allEmotionPrediction.index(i))

    emotion = []

    for j in guideEmotionMap:
        emotion.append(emotionMap[j]) 

    # All results
    return emotion, allEmotionPrediction

# # User variables

st.header("Let's use the model")

userText = st.text_area('Write something:')

#round(predictionSad[0][0]*100,2)

# Uso de función
if st.button('Predict'):

    emotion, allEmotionPrediction = predict_emotion(userText)

    # Interpretación de resultados
    st.write(f"📊 Emotion detected in the text:")
    for k in emotion:
        st.markdown(f"- {emotion[k]}")
    
    if st.button('More details'):

        # Definir un arreglo
        arreglo = allEmotionPrediction

        # Crear el histograma
        fig, ax = plt.subplots()
        ax.hist(arreglo, bins=6, edgecolor='black')

        # Añadir título y etiquetas
        ax.set_title('Emotion probabilities')
        ax.set_xlabel('Emotion')
        ax.set_ylabel('Probability')

        # Mostrar el histograma en la aplicación Streamlit
        st.pyplot(fig)