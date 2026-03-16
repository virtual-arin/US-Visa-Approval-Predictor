import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ---------------------------------
# App Configuration
# ---------------------------------
st.set_page_config(
    page_title="US Visa Approval Predictor",
    page_icon="🇺🇸",
    layout="centered"
)

st.title("US Visa Approval Predictor")
st.divider()

# ---------------------------------
# Load Model & Preprocessor
# ---------------------------------
@st.cache_resource
def load_objects():
    try:
        model = joblib.load("visa_model.pkl")
        preprocessor = joblib.load("preprocessor.pkl")
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, preprocessor = load_objects()

# ---------------------------------
# Input Section
# ---------------------------------
st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    continent = st.selectbox(
        "Continent of Origin",
        ['Asia', 'Africa', 'North America', 'Europe', 'South America', 'Oceania']
    )

    education_of_employee = st.selectbox(
        "Education Level",
        ["High School", "Bachelor's", "Master's", "Doctorate"]
    )

    has_job_experience = st.selectbox(
        "Has Job Experience?",
        ['Y', 'N']
    )

    requires_job_training = st.selectbox(
        "Requires Job Training?",
        ['Y', 'N']
    )

    region_of_employment = st.selectbox(
        "Region of Employment",
        ['West', 'Northeast', 'South', 'Midwest', 'Island']
    )

with col2:
    no_of_employees = st.number_input(
        "Number of Employees at Company",
        min_value=1,
        step=1,
        value=100
    )

    yr_of_estab = st.number_input(
        "Company Year of Establishment",
        min_value=1800,
        max_value=datetime.now().year,
        step=1,
        value=2000
    )

    prevailing_wage = st.number_input(
        "Prevailing Wage ($)",
        min_value=0.0,
        step=1000.0,
        value=60000.0
    )

    unit_of_wage = st.selectbox(
        "Unit of Wage",
        ['Hour', 'Week', 'Month', 'Year'],
        index=3
    )

    full_time_position = st.selectbox(
        "Full-Time Position?",
        ['Y', 'N']
    )

st.divider()

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("Predict Visa Status", type="primary", use_container_width=True):

    if model is None or preprocessor is None:
        st.error("Model or preprocessor failed to load.")
    else:

        # Create company_age feature (fix for missing column error)
        current_year = datetime.now().year
        company_age = current_year - yr_of_estab

        input_data = pd.DataFrame({
            'continent': [continent],
            'education_of_employee': [education_of_employee],
            'has_job_experience': [has_job_experience],
            'requires_job_training': [requires_job_training],
            'no_of_employees': [no_of_employees],
            'yr_of_estab': [yr_of_estab],
            'company_age': [company_age],   # FIXED
            'region_of_employment': [region_of_employment],
            'prevailing_wage': [prevailing_wage],
            'unit_of_wage': [unit_of_wage],
            'full_time_position': [full_time_position]
        })

        with st.spinner("Analyzing application data..."):

            try:
                # Preprocess input
                processed_input = preprocessor.transform(input_data)

                # Prediction
                prediction = model.predict(processed_input)

                # Probability (if model supports it)
                try:
                    probability = model.predict_proba(processed_input)[0][1]
                except:
                    probability = None

                result = "Certified" if prediction[0] == 1 else "Denied"

                if result == "Certified":
                    st.success(f"### Prediction: Visa Application **{result}** ✅")
                    st.balloons()
                else:
                    st.error(f"### Prediction: Visa Application **{result}** ❌")

                if probability is not None:
                    st.info(f"Approval Probability: **{probability:.2%}**")

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Ensure model and preprocessor match the training pipeline.")