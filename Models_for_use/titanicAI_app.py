import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Titanic survival algorithm", page_icon="🚢")

st.title("🚢 Titanic survival algorithm 🚢")

st.subheader("Made by: Arian Vazquez Fernandez")

st.write(
    "Welcome to this website where you can use my artificial intelligence to predict survival on the Titanic"
)

st.header("Database citation and use")

st.markdown("The Database that we will use is \"Titanic dataset\" from kaggle:"
            "(https://www.kaggle.com/datasets/brendan45774/test-file)"
)

# # Base de datos

# operation Github
@st.cache_data
def load_data():
    df = pd.read_csv("Download_Files/databaseUse/databaseTitanic.csv")
    model_loaded_joblib = joblib.load('Download_Files/modelsUse/titanicModel.pkl')
    ct = joblib.load('Download_Files/mathBoy/columnTransformerTitanic.pkl')
    return df, model_loaded_joblib, ct

df, model_loaded_joblib, ct = load_data()

st.write(df)

st.header("How to use it?")

st.write(
    "We must understand the variables involved in the algorithm that the user can place:"
)

st.markdown("""
- (Pclass): Refers to the most approximate socioeconomic position.
- (Sex): Refers to the sex of the passenger.
- (Age): Refers to the age of the passenger.
- (SibSp): It refers to the number of siblings or spouses of the passenger, this number does not distinguish between both groups.
- (Parch): It refers to the number of children or parents of the passenger, this number does not distinguish between both groups.
""")

st.write(
    "More details of the model in my repository on Github."
)

# Uso del bot
def testBotNewDatan(classChoosed,Sex,Age,SibSp=0,Parch=0):

    # Condicional Class
    FirstClass = 0
    SecondClass = 0
    ThirdClass = 0

    if classChoosed == 'First Class':
        FirstClass = 1
    elif classChoosed == 'Second Class':
        SecondClass = 1
    else:
        ThirdClass = 1

    # Conditional sexo
    if Sex == 'Female':
        Sex = 1
    else:
        Sex = 0

    X_new = np.array([[FirstClass, SecondClass, ThirdClass, Sex, Age, SibSp, Parch]])
    X_new = pd.DataFrame(X_new, columns=['FirstClass', 'SecondClass', 'ThirdClass', 'Sex', 'Age', 'SibSp', 'Parch'])
    X_new = ct.transform(X_new)

    y_estimate = model_loaded_joblib.predict(X_new, verbose=0)
    y_estimate_class = np.argmax(y_estimate, axis=1)[0]
    y_estimate_prob = y_estimate[0][y_estimate_class]

    return y_estimate, y_estimate_prob

# # User variables

st.header("Let's use the model")

st.write("⚠️ After entering a data, you must press the (Enter) key, and do not use the (+) and (-) buttons in the numerical boxes too much, as it may cause problems when loading the new data in streamlib. ⚠️")

optionClassPrediction = st.selectbox('Choose the class of the passenger:', ("First Class", "Second Class", "Third Class"))

optionGenderPrediction = st.selectbox('Choose the sex of the passenger:', ("Female", "Male"))

ageForPrediction = st.number_input('Enter your age:', min_value=1, max_value=150, value=21)

siblingsForPrediction = st.number_input('Enter number of siblings or spouses:', min_value=0, max_value=50, value=0) # Why 50?

parchForPrediction = st.number_input('number of children or parents:', min_value=0, max_value=50, value=0) # Very busy?

# Uso de función
if st.button('Predict'):

    resultado, probabilidad = testBotNewDatan(optionClassPrediction,optionGenderPrediction,ageForPrediction,siblingsForPrediction,parchForPrediction)

    # Interpretación de resultados
    st.write(f"📊 The probability of surviving is: {(resultado[0][1]*100):.2f}% 📊")