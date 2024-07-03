from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
import streamlit as st

# ---------------------------------------------------------------------------------------------------

# Cargar el dataset de Hugging Face
dataset = load_dataset("dair-ai/emotion")

# Mapa de emociones:
# ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
#      0        1       2       3       4          5
indexOfSentiment = 3

# Crear un conjunto de datos binario para la emoción concreta
def binary_label(example):
    example['label'] = 1 if example['label'] == indexOfSentiment else 0
    return example

train_dataset = dataset['train'].map(binary_label)

train_texts = train_dataset['text']

# Tokenización y Padding
max_features = 10000
max_len = 100

tok = Tokenizer(num_words=max_features)
tok.fit_on_texts(train_texts)

# ---------------------------------------------------------------------------------------------------

st.set_page_config(page_title="AI to predict feelings from text", page_icon="🧐")

st.title("AI to predict feelings from text 📚")

st.subheader("Made by: Arian Vazquez Fernandez")

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
    modelLoadedJoblibSurprise = load_model("Download_Files/modelsUse/arianMind/surpriseModelCritic.keras")
    modelLoadedJoblibSad = load_model('Download_Files/modelsUse/arianMind/sadnessModelCritic.keras')
    modelLoadedJoblibFear = load_model('Download_Files/modelsUse/arianMind/fearModelCritic.keras')
    modelLoadedJoblibJoy = load_model('Download_Files/modelsUse/arianMind/joyModelCritic.keras')
    modelLoadedJoblibLove = load_model('Download_Files/modelsUse/arianMind/loveModelCritic.keras')
    modelLoadedJoblibAnger = load_model('Download_Files/modelsUse/arianMind/angerModelCritic.keras')

    # Return    
    return modelLoadedJoblibSad, modelLoadedJoblibAnger, modelLoadedJoblibFear, modelLoadedJoblibJoy, modelLoadedJoblibLove, modelLoadedJoblibSurprise

modelLoadedJoblibSad, modelLoadedJoblibAnger, modelLoadedJoblibFear, modelLoadedJoblibJoy, modelLoadedJoblibLove, modelLoadedJoblibSurprise = load_data()

st.write(
    "More details of the model in my repository on Github."
)

# Función para predecir emociones
def preprocess_text(text):
    sequence = tok.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    return padded_sequence

# Predecir emoción
emotionMap = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
def predict_emotion(text):
    processed_text = preprocess_text(text)

    # Predicción modelos
    predictionSad = modelLoadedJoblibSad.predict(processed_text,verbose=0)
    predictionAnger = modelLoadedJoblibAnger.predict(processed_text,verbose=0)
    predictionFear = modelLoadedJoblibFear.predict(processed_text,verbose=0)
    predictionJoy = modelLoadedJoblibJoy.predict(processed_text,verbose=0)
    predictionLove = modelLoadedJoblibLove.predict(processed_text,verbose=0)
    predictionSurprise = modelLoadedJoblibSurprise.predict(processed_text,verbose=0)

    allEmotionPrediction = [predictionSad[0][0],predictionJoy[0][0],predictionLove[0][0],predictionAnger[0][0],predictionFear[0][0],predictionSurprise[0][0]]
    
    # Probabilidad de cada uno
    bestEmotion = max(allEmotionPrediction)
    guideEmotionMap = []
    allEmotionPredictionNor = []

    for i in allEmotionPrediction:
        if bestEmotion == i:
            guideEmotionMap.append(allEmotionPrediction.index(i))
            allEmotionPredictionNor.append(round(i*100,2))
        else:
            allEmotionPredictionNor.append(round(i*100,2))

    emotion = []

    for j in guideEmotionMap:
        emotion.append(emotionMap[j]) 

    # All results
    return emotion, allEmotionPredictionNor

# Example data
st.header("Data for validation (you can use it)")

# Mapa de emociones:
# ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
#      0        1       2       3       4          5

st.markdown("""
- Sadness  (0) 😞
- Joy      (1) 😄
- Love     (2) 😍
- Anger    (3) 😡
- Fear     (4) 😱
- Surprise (5) 🤯
""")

st.write(pd.DataFrame(dataset['validation']))

st.write('The evaluation with this database results in an accuracy of 90.25%, these 2000 rows of data have never been seen by the AI and are not used for training.')

# # User variables
st.header("Let's use the model")

st.write('⚠️ After you type something, you must press the (Ctrl) and (Enter) keys at the same time, this avoids errors with data loading in the streamlib application. 🤔')

userText = st.text_area('Write something:')

# Uso de función
if st.button('Predict'):

    emotion, allEmotionPredictionNor = predict_emotion(userText)
    
    # Interpretación de resultados
    st.write(f"📊 Emotion detected in the text: {emotion} 📊")

    # Crear el histograma
    fig, ax = plt.subplots()

    ax.bar(emotionMap, allEmotionPredictionNor, edgecolor='black')  # Crear el histograma de barras
    
    ax.set_xticks(emotionMap)  # Establecer las posiciones de las etiquetas
    ax.set_xticklabels(emotionMap, rotation=45)  # Asignar las etiquetas y rotarlas si es necesario
    
    st.pyplot(fig)  # Dibujar la figura actualizada