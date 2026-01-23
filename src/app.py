from flask import Flask, request, render_template, session, send_file
import os

from src.prediction.clinical_predictor import get_clinical_probability_from_form
from src.prediction.image_predictor import get_image_probability
from src.prediction.fusion_predictor import fuse_probabilities

from src.explainability.clinical_shap_utils import get_patient_shap_explanation, save_shap_plot
from src.explainability.gradcam_utils import generate_gradcam

app = Flask(__name__)
app.secret_key = "pcos_secret_key"

STATIC_EXPLAIN = os.path.join("static", "explainability")
os.makedirs(STATIC_EXPLAIN, exist_ok=True)


# =========================
# HOME
# =========================
@app.route("/")
def form():
    return render_template("form.html")

# Add these new routes to your existing app.py

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/screening")
def screening():
    return render_template("form.html")  # Same as home

@app.route("/resources")
def resources():
    return render_template("resources.html")

# Update the home route to use base template
@app.route("/")
def home():
    return render_template("form.html")  # Now uses base.html template
# =========================
# SUBMIT PIPELINE
# =========================
@app.route("/submit", methods=["POST"])
def submit():

    form_data = {
        "age": request.form.get("age"),
        "bmi": request.form.get("bmi"),
        "cycle_length": request.form.get("cycle_length"),
        "follicle_left": request.form.get("follicle_left"),
        "follicle_right": request.form.get("follicle_right"),
        "amh": request.form.get("amh"),
        "weight_gain": request.form.get("weight_gain")
    }

    image_file = request.files.get("ultrasound_image")
    os.makedirs("uploads", exist_ok=True)
    image_path = None

    if image_file:
        image_path = os.path.join("uploads", image_file.filename)
        image_file.save(image_path)

    # Predictions
    clinical_prob, input_vector, feature_names = get_clinical_probability_from_form(form_data)
    image_prob = get_image_probability(image_path)

    prediction, confidence, mode = fuse_probabilities(clinical_prob, image_prob)

    # ===== SHAP =====
    clinical_features, shap_values = get_patient_shap_explanation(
        input_vector, feature_names
    )

    shap_plot_path = os.path.join(
        app.root_path, "static", "explainability", "shap_plot.png"
    )
    save_shap_plot(shap_values, feature_names, shap_plot_path)

    # ===== GRADCAM =====
    gradcam_path = os.path.join(
        app.root_path, "static", "explainability", "gradcam_result.jpg"
    )
    generate_gradcam(image_path, gradcam_path)

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence,
        fusion_mode=mode,
        clinical_prob=round(clinical_prob, 3),
        image_prob=round(image_prob, 3),
        clinical_features=clinical_features,
        shap_image="/static/explainability/shap_plot.png",
        gradcam_image="/static/explainability/gradcam_result.jpg"
    )


# =========================
# DOWNLOAD REPORT
# =========================
@app.route("/download_report")
def download_report():

    content = f"""
PCOS Detection Report

Prediction: {session.get("prediction")}
Confidence: {session.get("confidence")}

Clinical Probability: {session.get("clinical_prob")}
Image Probability: {session.get("image_prob")}
Fusion Mode: {session.get("fusion_mode")}
"""

    report_path = "pcos_report.txt"
    with open(report_path, "w") as f:
        f.write(content)

    return send_file(report_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
