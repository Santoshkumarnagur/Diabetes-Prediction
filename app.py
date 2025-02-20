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

st.title("Diabetes Prediction using ML")

# Initialize session state for toggle button
if "show_accuracy" not in st.session_state:
    st.session_state.show_accuracy = False

# Toggle function
def toggle_accuracy():
    st.session_state.show_accuracy = not st.session_state.show_accuracy

# Button to toggle accuracy display
if st.button("Show/Hide Model Accuracy", on_click=toggle_accuracy):
    pass  # The function handles toggling

# Display accuracy if toggle is True
if st.session_state.show_accuracy:
    if accuracy:
        st.success(f"Model Accuracy: {accuracy*100:.2f}%")
    else:
        st.warning("Model accuracy not available.")

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

# Prediction button
if st.button("Predict"):
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
            st.error("The person is likely to have diabetes.")
        else:
            st.success("The person is not likely to have diabetes.")
    
    except ValueError:
        st.warning("Please enter valid numerical values for all inputs.")