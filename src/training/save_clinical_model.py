import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.data_preprocessing.clinical_preprocessing import load_and_prepare_clinical_data


def train_and_save_clinical_model():
    # 1. Load data
    X, y = load_and_prepare_clinical_data(
        "data/raw/clinical_without_infertility.csv"
    )

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 4. Create models folder if missing
    os.makedirs("models", exist_ok=True)

    # 5. Save model
    model_path = "models/clinical_rf.pkl"
    joblib.dump(model, model_path)

    print(f"Clinical model saved successfully at: {model_path}")


if __name__ == "__main__":
    train_and_save_clinical_model()
