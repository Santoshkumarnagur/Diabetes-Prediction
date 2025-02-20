import pickle
import numpy as np
import streamlit as st

# Load the trained model
with open("diabetes_model.sav", "rb") as file:
    model = pickle.load(file)

# Load accuracy score if available
try:
    with open("diabetes_accuracy.sav", "rb") as file:
        accuracy = pickle.load(file)
except FileNotFoundError:
    accuracy = None  # Handle case where accuracy is unavailable

st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🧑‍⚕")

# Custom CSS for styling
st.markdown("""
    <style>
    /* Stylish Title */
    .title-container {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #007BFF, #00C9FF);
        border-radius: 10px;
        color: white;
        font-size: 36px;
        font-weight: bold;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    
    /* Modern Buttons */
    .modern-button {
        background-color: #FF5733;
        color: white;
        border: none;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        cursor: pointer;
        transition: background 0.3s ease;
        width: 250px;
        display: block;
        margin: 10px auto;
        text-align: center;
    }
    .modern-button:hover {
        background-color: #E74C3C;
    }
    
    /* Predict Button */
    .predict-button {
        background-color: #28A745;
        color: white;
        border: none;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        cursor: pointer;
        transition: background 0.3s ease;
        width: 250px;
        display: block;
        margin: 10px auto;
        text-align: center;
    }
    .predict-button:hover {
        background-color: #218838;
    }
    </style>
""", unsafe_allow_html=True)

# Stylish Title
st.markdown('<div class="title-container">🩺 Diabetes Prediction using Machine Learning</div>', unsafe_allow_html=True)

# Initialize session state for toggle button
if "show_accuracy" not in st.session_state:
    st.session_state.show_accuracy = False

# Toggle function
def toggle_accuracy():
    st.session_state.show_accuracy = not st.session_state.show_accuracy

# Toggle button for accuracy
if st.button("📊 Show/Hide Model Accuracy", key="accuracy_button"):
    toggle_accuracy()

# Display accuracy if toggle is True
if st.session_state.show_accuracy:
    if accuracy:
        st.success(f"✅ Model Accuracy: *{accuracy*100:.2f}%*")
    else:
        st.warning("⚠ Model accuracy not available.")

# User input fields
col1, col2, col3 = st.columns(3)

with col1:
    Pregnancies = st.text_input("Number of Pregnancies", "0")

with col2:
    Glucose = st.text_input("Glucose Level", "0")

with col3:
    BloodPressure = st.text_input("Blood Pressure Value", "0")

with col1:
    SkinThickness = st.text_input("Skin Thickness Value", "0")

with col2:
    Insulin = st.text_input("Insulin Level", "0")

with col3:
    BMI = st.text_input("BMI Value", "0")

with col1:
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function", "0.0")

with col2:
    Age = st.text_input("Age of the Person", "0")

# Predict Button with Custom Color
predict_clicked = st.button("🔍 Predict", key="predict_button")

if predict_clicked:
    try:
        # Convert inputs to float
        input_data = np.array([
            float(Pregnancies), float(Glucose), float(BloodPressure), 
            float(SkinThickness), float(Insulin), float(BMI), 
            float(DiabetesPedigreeFunction), float(Age)
        ]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Display result
        if prediction[0] == 1:
            st.error("🚨 The person is *likely* to have diabetes.")
        else:
            st.success("✅ The person is *not likely* to have diabetes.")
    
    except ValueError:
        st.warning("⚠ Please enter valid *numerical* values for all inputs.")