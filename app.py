import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# TITLE
# -------------------------------
st.set_page_config(page_title="Crop Disease System", layout="wide")
st.title("🌾 Crop Disease & Pest Prediction Analytics System")

# -------------------------------
# MODEL
# -------------------------------
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
# FAKE WEATHER (NO API)
# -------------------------------
def get_fake_weather(city):
    np.random.seed(len(city))

    temp = np.random.randint(20, 40)
    humidity = np.random.randint(50, 95)
    rainfall = np.random.randint(0, 20)

    return temp, humidity, rainfall

# -------------------------------
# IMAGE MODEL
# -------------------------------
def predict_image(image):
    img = np.array(image)
    avg = img.mean()

    if avg < 100:
        return "Disease Detected"
    else:
        return "Healthy"

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("📊 Input Parameters")

city = st.sidebar.text_input("Enter Location", "Delhi")
crop = st.sidebar.selectbox("Select Crop", ["Rice", "Wheat", "Corn"])
stage = st.sidebar.selectbox("Growth Stage", ["Seedling", "Vegetative", "Flowering", "Harvest"])

# -------------------------------
# ANALYZE BUTTON
# -------------------------------
if st.sidebar.button("🔍 Analyze Risk"):

    temp, humidity, rainfall = get_fake_weather(city)

    st.subheader("🌦️ Weather Data (Demo)")
    col1, col2, col3 = st.columns(3)

    col1.metric("Temperature (°C)", temp)
    col2.metric("Humidity (%)", humidity)
    col3.metric("Rainfall (mm)", rainfall)

    # -------------------------------
    # DFI
    # -------------------------------
    dfi = (humidity * 0.5) + (rainfall * 0.3) + (temp * 0.2)

    st.subheader("🧠 Disease Favorability Index")
    st.write(round(dfi, 2))

    # -------------------------------
    # PREDICTION
    # -------------------------------
    prob = model.predict_proba([[temp, humidity, rainfall]])[0][1]

    st.subheader("⚠️ Disease Probability")
    st.write(round(prob, 2))

    # -------------------------------
    # RISK LEVEL
    # -------------------------------
    if prob < 0.3:
        st.success("Low Risk")
    elif prob < 0.7:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")
        st.write("Apply preventive spray")

    # -------------------------------
    # WHAT IF
    # -------------------------------
    st.subheader("🔮 What-if Analysis")

    new_prob = model.predict_proba([[temp, humidity, rainfall + 10]])[0][1]
    st.write("If rainfall increases → Risk:", round(new_prob, 2))

# -------------------------------
# IMAGE
# -------------------------------
st.subheader("📸 Leaf Image Analysis")

file = st.file_uploader("Upload Leaf Image")

if file:
    img = Image.open(file)
    st.image(img)

    result = predict_image(img)
    st.write("Result:", result)

# -------------------------------
# DASHBOARD
# -------------------------------
st.subheader("📊 Analytics Dashboard")

data = pd.read_csv("data.csv")

st.line_chart(data[["temperature", "humidity", "rainfall"]])
st.bar_chart(data["disease"].value_counts())

# -------------------------------
# FINAL
# -------------------------------
st.subheader("📢 Final Insight")
st.write("AI predicts disease before it happens")
