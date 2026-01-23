import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.data_preprocessing.clinical_preprocessing import load_and_prepare_clinical_data

# Load clinical data
X, y = load_and_prepare_clinical_data("data/raw/clinical_without_infertility.csv")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Random Forest (controlled complexity)
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

print("Training Clinical Random Forest...")
rf_model.fit(X_train, y_train)

# Probability predictions
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Save outputs
np.save("outputs/clinical_probabilities.npy", y_prob)
np.save("outputs/clinical_true_labels.npy", y_test.values)

print("Clinical RF probabilities saved.")
print("Probabilities shape:", y_prob.shape)

# Sanity check
print("\nClinical RF Classification Report:")
print(classification_report(y_test, rf_model.predict(X_test)))
