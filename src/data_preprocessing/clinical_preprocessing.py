import pandas as pd

def load_and_prepare_clinical_data(csv_path):
    df = pd.read_csv(csv_path)

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Clean column names
    df.columns = df.columns.str.strip()

    # Drop rows with missing target
    df = df.dropna(subset=["PCOS (Y/N)"])

    # Target
    y = df["PCOS (Y/N)"].astype(int)

    # Drop non-feature columns
    X = df.drop(
        columns=[
            "PCOS (Y/N)",
            "Sl. No",
            "Patient File No."
        ]
    )

    # Convert Y/N columns
    yn_columns = [col for col in X.columns if "(Y/N)" in col]
    for col in yn_columns:
        X[col] = X[col].map({"Y": 1, "N": 0})

    # Convert everything else to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # 🔴 DROP columns that are completely NaN
    X = X.dropna(axis=1, how="all")

    # Fill remaining NaNs with median
    X = X.fillna(X.median())

    return X, y
