# app.py - UPDATED FOR PYTHON 3.13+

# Standard library imports first
import pickle
from pathlib import Path
import os

# Third-party imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must come before pyplot import
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier  # or whatever model you're using


# Set page config
st.set_page_config(
    page_title="Hypertension Risk Predictor",
    page_icon="🫀",
    layout="wide"
)

# Title and description
st.title("🫀 Hypertension Risk Assessment AI")
st.markdown("""
This AI model predicts hypertension risk based on age, blood pressure readings, and pulse.
Enter your values below to get a personalized risk assessment.
""")

# Sidebar for input
st.sidebar.header("Patient Information")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model_path = Path(__file__).parent / "bp_models" / "bP_model.pickle"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'bp_models/bP_model.pickle' exists.")
        return None

model = load_model()

if model is not None:
    # Input fields
    age = st.sidebar.number_input("Age (years)", min_value=1, max_value=120, value=45, step=1)
    systolic = st.sidebar.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=120, step=1)
    diastolic = st.sidebar.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80, step=1)
    pulse = st.sidebar.number_input("Pulse Rate (bpm)", min_value=40, max_value=200, value=75, step=1)
    
    # Prediction function
    def predict_hypertension(Age, Systolic, Diastolic, Pulse):
        features = np.array([Age, Systolic, Diastolic, Pulse]).reshape(1, -1)
        prob = model.predict_proba(features)[0][1]
        prediction = model.predict(features)[0]
        risk_level = 'High' if prob > 0.7 else 'Moderate' if prob > 0.3 else 'Low'
        
        return {
            'probability': float(prob),
            'prediction': 'Hypertension' if prediction == 1 else 'Non-hypertensive',
            'risk_level': risk_level,
            'raw_prediction': prediction
        }
    
    def get_interpretation(prediction, prob, Systolic, Diastolic):
        if prediction == 1:
            if Systolic >= 160 or Diastolic >= 100:
                return "**Stage 2 Hypertension** - 🚨 Seek immediate medical attention"
            elif Systolic >= 140 or Diastolic >= 90:
                return "**Stage 1 Hypertension** - ⚠️ Consult your Doctor"
            else:
                return "**Elevated risk** based on multiple factors - 📊 Monitor regularly"
        else:
            if Systolic >= 130 or Diastolic >= 81:
                return "**Elevated blood pressure** - 👁️ Monitor regularly"
            else:
                return "**Normal blood pressure** - ✅ Maintain healthy lifestyle"
    
    # Predict button
    if st.sidebar.button("🔍 Assess Risk", type="primary"):
        with st.spinner("Analyzing your data..."):
            result = predict_hypertension(age, systolic, diastolic, pulse)
            interpretation = get_interpretation(
                result['raw_prediction'], 
                result['probability'], 
                systolic, 
                diastolic
            )
            
            # Display results in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Level", result['risk_level'])
                
            with col2:
                st.metric("Prediction", result['prediction'])
                
            with col3:
                st.metric("Probability", f"{result['probability']*100:.1f}%")
            
            # Progress bar for risk probability
            st.progress(result['probability'])
            
            # Interpretation box
            st.info(interpretation)
            
            # Detailed information
            with st.expander("📋 View Detailed Information"):
                st.write(f"""
                **Input Values:**
                - Age: {age} years
                - Systolic BP: {systolic} mmHg
                - Diastolic BP: {diastolic} mmHg
                - Pulse: {pulse} bpm
                
                **Risk Categories:**
                - Low Risk: < 30% probability
                - Moderate Risk: 30-70% probability
                - High Risk: > 70% probability
                
                **Note:** This is an AI prediction tool, not medical advice. 
                Always consult with a healthcare professional.
                """)
    
    # BMI Calculator
    st.sidebar.markdown("---")
    st.sidebar.subheader("BMI Calculator")
    height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170, step=1)
    weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70, step=1)
    
    if height > 0:
        #bmi = weight / (height ** 2) # height in meters
        bmi = weight / ((height/100) ** 2)
        st.sidebar.metric("BMI", f"{bmi:.1f}")
        
        if bmi < 18.5:
            st.sidebar.caption("Underweight")
        elif bmi < 25:
            st.sidebar.caption("Normal weight")
        elif bmi < 30:
            st.sidebar.caption("Overweight")
        else:
            st.sidebar.caption("Obese")
    
    # Main area info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Normal Blood Pressure Ranges")
        st.markdown("""
        - **Normal:** < 120/80 mmHg
        - **Elevated:** 120-129/<80 mmHg
        - **Hypertension Stage 1:** 130-139/80-89 mmHg
        - **Hypertension Stage 2:** ≥140/≥90 mmHg
        """)
    
    with col2:
        st.subheader("⚠️ Disclaimer")
        st.markdown("""
        This tool provides AI-based predictions for educational purposes only.
        - Not a substitute for professional medical advice
        - Accuracy: 90% on test data
        - Always consult healthcare providers
        """)
    
    # Footer
    st.markdown("---")
    st.caption("AI Model: Random Forest Classifier | Data: Clinical BP Measurements")
    
else:

    st.error("Unable to load the prediction model. Please check the model file.")

