import shap
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import joblib

model = joblib.load("models/clinical_rf.pkl")

explainer = shap.TreeExplainer(model)


def get_patient_shap_explanation(input_vector, feature_names, top_k=3):

    shap_values = explainer.shap_values(input_vector)[1][0]

    impacts = list(zip(feature_names, shap_values))
    impacts = sorted(impacts, key=lambda x: abs(x[1]), reverse=True)

    explanation = []
    for f, v in impacts[:top_k]:
        direction = "increases" if v > 0 else "decreases"
        explanation.append(f"{f} → {direction} risk")

    return explanation, shap_values



def save_shap_plot(shap_values, feature_names, output_path, top_k=6):

    import numpy as np
    import matplotlib.pyplot as plt
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    shap_values = np.array(shap_values).flatten()

    # Select top K features by absolute impact
    indices = np.argsort(np.abs(shap_values))[-top_k:]
    selected_values = shap_values[indices]
    selected_features = [feature_names[i] for i in indices]

    plt.figure(figsize=(7,5))
    colors = ["red" if v > 0 else "blue" for v in selected_values]

    plt.barh(selected_features, selected_values, color=colors)
    plt.xlabel("SHAP Impact")
    plt.title("Top Clinical Feature Contributions")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

