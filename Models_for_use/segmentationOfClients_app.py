import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Customer Segmentation Tool", layout="wide")
st.title("Customer Segmentation Tool with K-means")

st.write('''The database we will use is ["Mall Customer Segmentation Data"](https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fpolarbearwyy%2Fmall-customer-segmentation-data) 
         from "Kaggle". If you want to know the results of the analysis, please check the PDF file in the left sidebar of this page.''')

# Data loading
@st.cache_data
def load_data():

    # Our data
    df = pd.read_csv("C:/Users/arian/OneDrive/Documentos/GitHub/Artificial-Intelligence-Resources-New/Download_Files/databaseUse/KMeanCustomerSegmentation.csv")
    
    # Delete "CustomerID" column
    df.drop(['CustomerID'], axis=1, inplace=True)
    
    return df

data = load_data()

# Display the data
st.subheader("Customer Data")
st.write(data.head())

# Create a "Gender" column with numeric values
data['Gender'] = data['Gender'].apply(lambda x: 1 if (x == 'Female' or x == 1) else 0)

# Feature selection for segmentation
st.sidebar.header("Segmentation Configuration")
features = st.sidebar.multiselect(
    "Select features for segmentation",
    options=data.columns,
    default=data.columns.tolist()
)

# Number of clusters selection
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)

# Explanation for the user
st.sidebar.write("Click the button below to download the PDF file and know the results of the analysis.")

# PDF file path
pdf_file_path_en = "Download_Files/databaseUse/CustomerSegmentationPDF/Customer segmentation analysis results.pdf"
pdf_file_path_es = "Download_Files/databaseUse/CustomerSegmentationPDF/Resultados del análisis de segmentación de clientes.pdf"

# Read the PDF file as binary
with open(pdf_file_path_en, "rb") as pdf_file:
    pdf_bytes_en = pdf_file.read()
with open(pdf_file_path_es, "rb") as pdf_file:
    pdf_bytes_es = pdf_file.read()

# Download button
st.sidebar.download_button(
    label="Download PDF (English)",
    data=pdf_bytes_en,
    file_name="Customer segmentation analysis results.pdf",
    mime="application/pdf"
)

st.sidebar.download_button(
    label="Descargar PDF (Español)",
    data=pdf_bytes_es,
    file_name="Resultados del análisis de segmentación de clientes.pdf",
    mime="application/pdf"
)

# Check which features should be standardized
def check_standardization(features, data):
    to_standardize = []
    for feature in features:
        if data[feature].dtype in ['int64', 'float64']:
            if data[feature].min() < 0 or data[feature].max() > 1:
                to_standardize.append(feature)
    return to_standardize

# Identify features that may need standardization
features_to_standardize = check_standardization(features, data)

if features_to_standardize:
    st.warning(f"The following features should be standardized: {', '.join(features_to_standardize)}")

# Preprocessing and K-means application
try:
    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X)

    # Results visualization
    st.subheader(f"Segmentation Results (K={n_clusters})")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(data[features[0]], data[features[1]], c=data['Cluster'], cmap='viridis')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title(f"Customer Segmentation: {features[0]} vs {features[1]}")
    plt.colorbar(scatter)
    st.pyplot(fig)

    # Cluster statistics
    st.subheader("Cluster Statistics")
    cluster_stats = data.groupby('Cluster')[features].mean()
    st.write(cluster_stats)

    # Additional plots
    st.subheader("Additional Plots")

    # Histogram
    feature_to_plot = st.selectbox("Select a feature for the histogram", options=features)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_clusters):
        sns.histplot(data[data['Cluster'] == i][feature_to_plot], kde=True, label=f'Cluster {i}', ax=ax)
    ax.set_title(f"Distribution of {feature_to_plot} by Cluster")
    ax.legend()
    st.pyplot(fig)

    # Box plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Cluster', y=feature_to_plot, data=data, ax=ax)
    ax.set_title(f"Distribution of {feature_to_plot} by Cluster")
    st.pyplot(fig)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    corr_matrix = data[features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

except Exception as e:
    st.warning(f"Waiting for at least 2 features for segmentation. Error: {str(e)}")
