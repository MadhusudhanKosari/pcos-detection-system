import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def train_and_save_image_model():
    # 1. Load extracted features
    X = np.load("data/processed/image_features.npy")
    y = np.load("data/processed/image_labels.npy")

    print("Loaded image features:", X.shape)
    print("Loaded image labels:", y.shape)

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. Train Random Forest
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # 4. Evaluate model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

    # 5. Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/image_rf_classifier.pkl"
    joblib.dump(model, model_path)

    print(f"\nImage RF model saved at: {model_path}")


if __name__ == "__main__":
    train_and_save_image_model()
