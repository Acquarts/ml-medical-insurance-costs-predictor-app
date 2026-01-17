"""
Medical Insurance Cost Predictor - Streamlit App
Author: Adrián Zambrana
Description: Interactive web application to predict medical insurance costs
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from google.cloud import aiplatform
from google.cloud import storage

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #1565C0;
    }
    .info-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model from local file or GCS"""
    model_path = os.getenv("MODEL_PATH", "model/model.joblib")

    # Try to load from local path first
    if os.path.exists(model_path):
        return joblib.load(model_path)

    # Try to load from GCS
    gcs_bucket = os.getenv("GCS_BUCKET")
    gcs_model_path = os.getenv("GCS_MODEL_PATH", "models/insurance_model.joblib")

    if gcs_bucket:
        try:
            client = storage.Client()
            bucket = client.bucket(gcs_bucket)
            blob = bucket.blob(gcs_model_path)

            local_path = "/tmp/model.joblib"
            blob.download_to_filename(local_path)
            return joblib.load(local_path)
        except Exception as e:
            st.error(f"Error loading model from GCS: {e}")
            return None

    return None


def predict_with_vertex_ai(features: dict) -> float:
    """Make prediction using Vertex AI endpoint"""
    endpoint_id = os.getenv("VERTEX_ENDPOINT_ID")
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION", "us-central1")

    if not endpoint_id or not project_id:
        return None

    try:
        aiplatform.init(project=project_id, location=location)
        endpoint = aiplatform.Endpoint(endpoint_id)

        # Prepare instance for prediction
        instance = [list(features.values())]
        prediction = endpoint.predict(instances=instance)

        return prediction.predictions[0]
    except Exception as e:
        st.error(f"Error with Vertex AI prediction: {e}")
        return None


def predict_local(model, features: dict) -> float:
    """Make prediction using local model"""
    df = pd.DataFrame([features])
    return model.predict(df)[0]


def main():
    st.markdown('<h1 class="main-header">🏥 Medical Insurance Cost Predictor</h1>', unsafe_allow_html=True)

    st.markdown("""
    This application predicts annual medical insurance costs based on personal characteristics.
    The model was trained using Gradient Boosting with **90% accuracy**.
    """)

    # Sidebar for inputs
    st.sidebar.header("📋 Enter Your Information")

    # Age input
    age = st.sidebar.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        help="Your current age"
    )

    # Sex input
    sex = st.sidebar.selectbox(
        "Sex",
        options=["Female", "Male"],
        help="Biological sex"
    )

    # BMI input
    bmi = st.sidebar.slider(
        "BMI (Body Mass Index)",
        min_value=15.0,
        max_value=55.0,
        value=25.0,
        step=0.1,
        help="Body Mass Index (weight in kg / height in m²)"
    )

    # BMI category display
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal weight"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"
    st.sidebar.info(f"BMI Category: **{bmi_category}**")

    # Children input
    children = st.sidebar.selectbox(
        "Number of Children",
        options=[0, 1, 2, 3, 4, 5],
        help="Number of dependents covered by insurance"
    )

    # Smoker input
    smoker = st.sidebar.selectbox(
        "Smoking Status",
        options=["No", "Yes"],
        help="Do you smoke?"
    )

    # Region input
    region = st.sidebar.selectbox(
        "Region",
        options=["Northeast", "Northwest", "Southeast", "Southwest"],
        help="Your residential area in the US"
    )

    # Prepare features for prediction
    features = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "Male" else 0,
        "smoker_yes": 1 if smoker == "Yes" else 0,
        "region_northeast": 1 if region == "Northeast" else 0,
        "region_northwest": 1 if region == "Northwest" else 0,
        "region_southeast": 1 if region == "Southeast" else 0,
        "region_southwest": 1 if region == "Southwest" else 0
    }

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Your Profile Summary")

        profile_data = {
            "Attribute": ["Age", "Sex", "BMI", "Children", "Smoker", "Region"],
            "Value": [f"{age} years", sex, f"{bmi:.1f} ({bmi_category})",
                     str(children), smoker, region]
        }
        st.table(pd.DataFrame(profile_data))

    with col2:
        st.subheader("⚠️ Risk Factors")

        risk_score = 0
        risks = []

        if smoker == "Yes":
            risk_score += 3
            risks.append("🚬 Smoking (HIGH impact)")
        if bmi >= 30:
            risk_score += 2
            risks.append("⚖️ High BMI (MEDIUM impact)")
        if age >= 50:
            risk_score += 1
            risks.append("📅 Age 50+ (LOW-MEDIUM impact)")

        if risks:
            for risk in risks:
                st.warning(risk)
        else:
            st.success("✅ No major risk factors detected!")

    # Prediction button
    st.markdown("---")

    if st.button("🔮 Predict Insurance Cost", type="primary", use_container_width=True):
        with st.spinner("Calculating prediction..."):
            # Try Vertex AI first, then local model
            prediction = predict_with_vertex_ai(features)

            if prediction is None:
                model = load_model()
                if model:
                    prediction = predict_local(model, features)
                else:
                    # Fallback: use a simple estimation based on the notebook analysis
                    st.warning("Model not loaded. Using estimation based on analysis.")
                    base_cost = 8000
                    prediction = base_cost
                    prediction += age * 250
                    prediction += bmi * 300
                    prediction += children * 500
                    if smoker == "Yes":
                        prediction *= 3.5

            if prediction:
                st.markdown(f"""
                <div class="prediction-box">
                    <p>Estimated Annual Insurance Cost</p>
                    <p class="prediction-value">${prediction:,.2f}</p>
                </div>
                """, unsafe_allow_html=True)

                # Additional insights
                st.subheader("💡 Key Insights")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Monthly Cost",
                        f"${prediction/12:,.2f}",
                        help="Estimated monthly premium"
                    )

                with col2:
                    avg_cost = 13270  # Average from the dataset
                    diff = ((prediction - avg_cost) / avg_cost) * 100
                    st.metric(
                        "vs. Average",
                        f"{diff:+.1f}%",
                        delta=f"${prediction - avg_cost:,.0f}",
                        help="Compared to dataset average ($13,270)"
                    )

                with col3:
                    if smoker == "Yes":
                        non_smoker_est = prediction / 3.5
                        savings = prediction - non_smoker_est
                        st.metric(
                            "Potential Savings",
                            f"${savings:,.0f}",
                            help="If you quit smoking"
                        )
                    else:
                        st.metric(
                            "Smoker Premium",
                            f"${prediction * 3.5:,.0f}",
                            help="What you'd pay if smoking"
                        )

    # Footer with model information
    st.markdown("---")
    st.markdown("""
    ### 📈 About the Model

    This prediction model was built using **Gradient Boosting Regressor** and trained on
    medical insurance data from 1,337 individuals.

    **Model Performance:**
    - R² Score: 0.90 (90% accuracy)
    - Mean Absolute Error: $2,530
    - Root Mean Square Error: $4,269

    **Key Factors (by importance):**
    1. 🚬 Smoking status (~70%)
    2. ⚖️ BMI (~15%)
    3. 📅 Age (~10%)
    4. 📍 Other factors (~5%)

    ---
    *Created by Adrián Zambrana | Deployed on Google Cloud Run*
    """)


if __name__ == "__main__":
    main()
