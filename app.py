import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Page Title
# ----------------------------

st.set_page_config(page_title="DDoS Attack Detection", layout="wide")

st.title("🚨 DDoS Attack Detection System")
st.write("Upload network traffic data to detect potential DDoS attacks using Machine Learning models.")

# ----------------------------
# Load Trained Models
# ----------------------------

@st.cache_resource
def load_models():
    with open("ddos_models.pkl", "rb") as f:
        models = pickle.load(f)
    return models

models = load_models()

rf_model = models["random_forest"]
lr_model = models["logistic_regression"]
nn_model = models["neural_network"]

# ----------------------------
# Model Selection
# ----------------------------

model_choice = st.selectbox(
    "Select Machine Learning Model",
    ("Random Forest", "Logistic Regression", "Neural Network")
)

# ----------------------------
# File Upload
# ----------------------------

uploaded_file = st.file_uploader(
    "Upload Network Traffic CSV File",
    type=["csv"]
)

# ----------------------------
# Prediction Section
# ----------------------------

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(data.head())

        st.write(f"Dataset Shape: {data.shape}")

        if st.button("Run DDoS Detection"):

            if model_choice == "Random Forest":
                predictions = rf_model.predict(data)

            elif model_choice == "Logistic Regression":
                predictions = lr_model.predict(data)

            else:
                predictions = nn_model.predict(data)

            # Add predictions to dataframe
            data["Prediction"] = predictions

            # Convert numeric prediction to label
            data["Prediction"] = data["Prediction"].map({
                0: "Normal Traffic",
                1: "DDoS Attack"
            })

            st.subheader("Prediction Results")
            st.dataframe(data)

            # Attack summary
            attack_count = (data["Prediction"] == "DDoS Attack").sum()
            normal_count = (data["Prediction"] == "Normal Traffic").sum()

            st.subheader("Traffic Summary")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Normal Traffic", normal_count)

            with col2:
                st.metric("DDoS Attacks Detected", attack_count)

    except Exception as e:
        st.error(f"Error processing file: {e}")

# ----------------------------
# Instructions
# ----------------------------

st.sidebar.header("Instructions")

st.sidebar.write(
"""
1️⃣ Upload a CSV file containing the **77 network traffic features**.

2️⃣ Select the machine learning model.

3️⃣ Click **Run DDoS Detection** to analyze the traffic.

The system will classify each network flow as:

- **Normal Traffic**
- **DDoS Attack**
"""
)