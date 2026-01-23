from PIL import Image
import os

from src.explainability.clinical_shap_utils import get_top_clinical_features


def show_xai_output():
    prediction = "PCOS"
    confidence = 0.84

    # 🔥 REAL SHAP FEATURES
    clinical_features = get_top_clinical_features(top_k=3)

    image_explanation_text = (
        "The model focused on ovarian regions and follicle distribution patterns."
    )

    image_path = "static/explainability/gradcam_example.jpg"

    print("\nPrediction Result")
    print("-----------------")
    print("Prediction:", prediction)
    print("Confidence:", confidence)

    print("\nClinical Explanation (SHAP-based)")
    print("--------------------------------")
    for f in clinical_features:
        print("-", f)

    print("\nImage Explanation")
    print("----------------")
    print(image_explanation_text)

    if os.path.exists(image_path):
        print("\nOpening image explanation...")
        img = Image.open(image_path)
        img.show()
    else:
        print("\nImage explanation file not found.")


if __name__ == "__main__":
    show_xai_output()
