import joblib
import numpy as np

MODEL_PATH = "models/clinical_rf.pkl"


def get_clinical_probability_from_form(form_data):
    """
    Takes form data dict and returns PCOS probability from clinical RF model.
    """

    # Load model
    model = joblib.load(MODEL_PATH)

    # Get feature order used during training
    feature_names = model.feature_names_in_

    # Initialize input with zeros
    input_vector = np.zeros(len(feature_names))

    # Map form fields to feature names
    feature_mapping = {
        "age": "Age (yrs)",
        "bmi": "BMI",
        "cycle_length": "Cycle length(days)",
        "follicle_left": "Follicle No. (L)",
        "follicle_right": "Follicle No. (R)",
        "amh": "AMH(ng/mL)",
        "weight_gain": "Weight gain(Y/N)"
    }

    # Fill values
    for form_key, feature_name in feature_mapping.items():
        if feature_name in feature_names:
            idx = list(feature_names).index(feature_name)
            input_vector[idx] = float(form_data[form_key])

    # Reshape for prediction
    input_vector = input_vector.reshape(1, -1)

    # Predict probability for PCOS
    prob = model.predict_proba(input_vector)[0][1]

    return round(float(prob), 3), input_vector, feature_names

