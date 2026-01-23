import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load image features
X = np.load("outputs/image_features.npy")
y = np.load("outputs/image_labels.npy")

# Train-test split (image-level, same ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Random Forest (same settings)
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

print("Training Image Random Forest...")
rf_model.fit(X_train, y_train)

# Probability predictions
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Save outputs
np.save("outputs/image_probabilities.npy", y_prob)
np.save("outputs/image_true_labels.npy", y_test)

print("Image RF probabilities saved.")
print("Probabilities shape:", y_prob.shape)

# Sanity check
print("\nImage RF Classification Report:")
print(classification_report(y_test, rf_model.predict(X_test)))
