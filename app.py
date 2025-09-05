import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --- PATH SETTINGS ---
# Build paths inside the project like this: BASE_DIR / 'filename'
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "breast_cancer_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# --- CONSTANTS ---
FEATURE_NAMES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
]

# Example default values (replace with actual mean values from your training data for better results)
DEFAULT_VALUES = {
    "radius_mean": 14.12,
    "texture_mean": 19.28,
    "perimeter_mean": 91.96,
    "area_mean": 654.88,
    "smoothness_mean": 0.096,
}

# --- LOAD SAVED MODEL & SCALER ---
@st.cache_resource
def load_assets(model_path, scaler_path):
    """Loads the pre-trained model and scaler from disk."""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        st.error(  # This will still display the error message
            f"Model file not found at {model_path} or scaler file not found at {scaler_path}. "
            "Please ensure the files are in the correct directory."
        )
        st.stop()  # This will halt the script execution immediately

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="centered"
)

# --- HELPER FUNCTIONS ---
def user_input_features():
    """Creates sidebar inputs for user data and returns them as a DataFrame."""
    st.sidebar.header("Patient Details")
    data = {}
    for feature in FEATURE_NAMES:
        # Create a more user-friendly label from the feature name
        label = feature.replace('_', ' ').title()
        default_val = DEFAULT_VALUES.get(feature, 0.0)
        data[feature] = st.sidebar.number_input(
            label,
            min_value=0.0,
            value=default_val,  # Set the default value here
            format="%.4f",
            key=feature)

    features = pd.DataFrame(data, index=[0], columns=FEATURE_NAMES)
    return features


# --- MAIN APPLICATION ---
def main():
    """Main function to run the Streamlit app."""
    # Load assets at the start of the main app execution
    model, scaler = load_assets(MODEL_PATH, SCALER_PATH)

    st.title("🩺 Breast Cancer Detection App")
    st.write("This app predicts whether a breast tumor is Benign or Malignant based on patient data.")
    st.write("Please enter the patient's details in the sidebar.")

    # --- HOW IT WORKS EXPANDER ---
    with st.expander("ℹ️ How It Works & Feature Descriptions"):
        st.markdown("""
            This application uses a pre-trained Machine Learning model (Logistic Regression) to predict breast cancer.
            - **Enter Data**: Use the sidebar to input the mean values for the tumor's key characteristics.
            - **Predict**: Click the 'Predict' button.
            - **Result**: The model will classify the tumor as either **Benign** (non-cancerous) or **Malignant** (cancerous) and show the prediction probabilities.
        """)

    input_df = user_input_features()

    # --- DISPLAY USER INPUT ---
    st.subheader("Patient Data Entered")
    st.write("Please review the data you entered in the sidebar.")
    st.dataframe(input_df)

    # --- PREDICTION LOGIC ---
    if st.sidebar.button("Predict"):
        if model is not None and scaler is not None:
            # Check if all inputs are at their default zero-value, which might be unintentional
            is_input_default = all(input_df[col][0] == 0.0 for col in FEATURE_NAMES)
            if is_input_default:
                st.warning("Please enter patient data in the sidebar before predicting.", icon="⚠️")
                return # Stop execution if no data is entered

            try:
                # Scale input and predict
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)

                st.subheader("🔬 Prediction Analysis")
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.metric(
                        label="Prediction",
                        value="Malignant" if prediction == 1 else "Benign",
                        delta="High Risk" if prediction == 1 else "Low Risk",
                        delta_color="inverse" if prediction == 1 else "normal"
                    )

                with col2:
                    st.write("Prediction Probability:")
                    prob_df = pd.DataFrame(prediction_proba, columns=['Benign', 'Malignant'])
                    st.bar_chart(prob_df)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Model and/or scaler not loaded. Cannot perform prediction.")

    # Add a reset button at the bottom of the sidebar
    if st.sidebar.button("Reset Inputs"):
        # A simple way to reset is to trigger a rerun, Streamlit will use the default values
        st.rerun()

    # --- FOOTER ---
    st.markdown("---")
    st.markdown(
        '<h6>Made with &hearts; by <a href="https://github.com/visd2">visd2</a> | '
        '<a href="https://github.com/visd2/breast_cancer_detetion">Source Code</a></h6>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
