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
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Default values are the mean values from the training dataset (scikit-learn's breast cancer dataset),
# which provide a sensible starting point for the user.
DEFAULT_VALUES = {
    "radius_mean": 14.127, "texture_mean": 19.290, "perimeter_mean": 91.969,
    "area_mean": 654.889, "smoothness_mean": 0.096, "compactness_mean": 0.104,
    "concavity_mean": 0.089, "concave points_mean": 0.049, "symmetry_mean": 0.181,
    "fractal_dimension_mean": 0.063, "radius_se": 0.405, "texture_se": 1.217,
    "perimeter_se": 2.866, "area_se": 40.337, "smoothness_se": 0.007,
    "compactness_se": 0.025, "concavity_se": 0.032, "concave points_se": 0.012,
    "symmetry_se": 0.021, "fractal_dimension_se": 0.004, "radius_worst": 16.269,
    "texture_worst": 25.677, "perimeter_worst": 107.261, "area_worst": 880.583,
    "smoothness_worst": 0.132, "compactness_worst": 0.254, "concavity_worst": 0.272,
    "concave points_worst": 0.115, "symmetry_worst": 0.290, "fractal_dimension_worst": 0.084
}

# --- LOAD SAVED MODEL & SCALER ---
@st.cache_resource
def load_assets(model_path, scaler_path):
    """
    Loads the pre-trained model and scaler from disk.
    Returns (None, None) if any asset is not found or fails to load.
    """
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        return None, None

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
    st.sidebar.write("Use the expanders to enter patient data.")
    data = {}
    # Group features for a better UI
    feature_groups = {
        "Mean Values": [f for f in FEATURE_NAMES if f.endswith('_mean')],
        "Standard Error Values": [f for f in FEATURE_NAMES if f.endswith('_se')],
        "Worst (Largest) Values": [f for f in FEATURE_NAMES if f.endswith('_worst')],
    }

    for group_name, features_in_group in feature_groups.items():
        with st.sidebar.expander(group_name):
            for feature in features_in_group:
                label = feature.replace('_', ' ').title()
                default_val = DEFAULT_VALUES.get(feature, 0.0)
                data[feature] = st.number_input(
                    label,
                    min_value=0.0,
                    value=default_val,
                    format="%.4f",
                    key=feature
                )

    # Ensure the DataFrame columns are in the original, correct order
    features_df = pd.DataFrame(data, index=[0])[FEATURE_NAMES]
    return features_df


# --- MAIN APPLICATION ---
def main():
    """Main function to run the Streamlit app."""
    # Load assets at the start of the main app execution
    model, scaler = load_assets(MODEL_PATH, SCALER_PATH)

    # Gracefully handle missing model/scaler files
    if model is None or scaler is None:
        st.error(f"Model or scaler files not found. Please ensure `{MODEL_PATH.name}` and `{SCALER_PATH.name}` are in the same directory as your Streamlit app.")
        st.stop()

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
        # Check if the user has not changed any of the default input values.
        default_series = pd.Series(DEFAULT_VALUES, index=FEATURE_NAMES)
        # Round to 4 decimal places to avoid floating point comparison issues
        if input_df.iloc[0].round(4).equals(default_series.round(4)):
            st.warning("Please enter patient data in the sidebar before predicting. The current values are defaults.", icon="⚠️")
            return  # Stop execution if no data is entered

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
