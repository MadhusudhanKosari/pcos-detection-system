import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.data_preprocessing.clinical_preprocessing import load_and_prepare_clinical_data

# Load data
X, y = load_and_prepare_clinical_data("data/raw/clinical_without_infertility.csv")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train RF
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# SHAP explainer
explainer = shap.TreeExplainer(model)

# IMPORTANT: use predict_proba
shap_values = explainer.shap_values(X_test)

# For binary classification, take class 1
if isinstance(shap_values, list):
    shap_values = shap_values[1]

print("SHAP values shape:", shap_values.shape)
print("X_test shape:", X_test.shape)

# Plot
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=X.columns,
    show=True
)
