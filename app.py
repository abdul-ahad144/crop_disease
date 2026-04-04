import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import os
from PIL import Image

# -------------------------------
# CONFIGURATION
# -------------------------------
st.set_page_config(page_title="PragyanAI Crop Analytics", layout="wide")

st.title("🌾 Crop Disease & Pest Prediction Analytics System")

# -------------------------------
# LOAD MODEL (AUTO TRAIN IF NOT EXIST)
# -------------------------------
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE = "model.pkl"

def train_model():
    data = pd.read_csv("data.csv")
    X = data[["temperature", "humidity", "rainfall"]]
    y = data["disease"]

    model = RandomForestClassifier()
    model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    return model

def load_model():
    if not os.path.exists(MODEL_FILE):
        return train_model()
    return pickle.load(open(MODEL_FILE, "rb"))

model = load_model()

# -------------------------------
# WEATHER FUNCTION
# -------------------------------
API_KEY = "YOUR_API_KEY"

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    res = requests.get(url).json()

    temp = res["main"]["temp"]
    humidity = res["main"]["humidity"]
    rainfall = res.get("rain", {}).get("1h", 0)

    return temp, humidity, rainfall

# -------------------------------
# IMAGE PREDICTION (BASIC)
# -------------------------------
def predict_image(image):
    img = np.array(image)
    avg = img.mean()

    if avg < 100:
        return "Disease Detected"
    else:
        return "Healthy"

# -------------------------------
# SIDEBAR INPUT
# -------------------------------
st.sidebar.header("📊 Input Parameters")

city = st.sidebar.text_input("Enter Location", "Delhi")
crop = st.sidebar.selectbox("Select Crop", ["Rice", "Wheat", "Corn"])
stage = st.sidebar.selectbox("Growth Stage", ["Seedling", "Vegetative", "Flowering", "Harvest"])

# -------------------------------
# WEATHER + PREDICTION
# -------------------------------
if st.sidebar.button("🔍 Analyze Risk"):

    try:
        temp, humidity, rainfall = get_weather(city)

        st.subheader("🌦️ Weather Data")
        col1, col2, col3 = st.columns(3)

        col1.metric("Temperature (°C)", temp)
        col2.metric("Humidity (%)", humidity)
        col3.metric("Rainfall (mm)", rainfall)

        # -------------------------------
        # FEATURE ENGINEERING (DFI)
        # -------------------------------
        dfi = (humidity * 0.5) + (rainfall * 0.3) + (temp * 0.2)

        st.subheader("🧠 Disease Favorability Index (DFI)")
        st.write(round(dfi, 2))

        # -------------------------------
        # PREDICTION
        # -------------------------------
        prob = model.predict_proba([[temp, humidity, rainfall]])[0][1]

        st.subheader("⚠️ Disease Probability Score")
        st.write(round(prob, 2))

        # -------------------------------
        # RISK CLASSIFICATION
        # -------------------------------
        st.subheader("🚨 Risk Level")

        if prob < 0.3:
            st.success("Low Risk")
        elif prob < 0.7:
            st.warning("Medium Risk")
        else:
            st.error("High Risk")

            st.subheader("💊 Recommendation Engine")
            st.write("Apply preventive fungicide within 2–3 days")

        # -------------------------------
        # WHAT-IF ANALYSIS
        # -------------------------------
        st.subheader("🔮 What-if Analysis")

        increased_rain = rainfall + 10
        new_prob = model.predict_proba([[temp, humidity, increased_rain]])[0][1]

        st.write("If rainfall increases → Risk becomes:", round(new_prob, 2))

    except Exception as e:
        st.error(f"Error: {e}")

# -------------------------------
# IMAGE ANALYSIS
# -------------------------------
st.subheader("📸 Leaf Image Analysis")

file = st.file_uploader("Upload Leaf Image", type=["jpg", "png"])

if file:
    img = Image.open(file)
    st.image(img, use_column_width=True)

    result = predict_image(img)

    st.subheader("🧪 Image Result")
    st.write(result)

# -------------------------------
# ANALYTICS DASHBOARD
# -------------------------------
st.subheader("📈 Analytics Dashboard")

data = pd.read_csv("data.csv")

st.write("### Weather vs Disease Correlation")
st.line_chart(data[["temperature", "humidity", "rainfall"]])

st.write("### Disease Distribution")
st.bar_chart(data["disease"].value_counts())

# -------------------------------
# FINAL OUTPUT MESSAGE
# -------------------------------
st.subheader("📢 Final Insight")

st.write("AI predicts disease before it happens using weather + crop + patterns")
