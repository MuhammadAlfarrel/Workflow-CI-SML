import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train():
    print("=== Training MLflow Project ===")

    # Pastikan dataset ada
    if not os.path.exists("diabetes_clean.csv"):
        raise FileNotFoundError("diabetes_clean.csv tidak ditemukan")

    # Load data
    df = pd.read_csv("diabetes_clean.csv")
    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", acc)

        # PENTING: artifact_path HARUS "model"
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        print("Training selesai")
        print(f"Accuracy: {acc}")

if __name__ == "__main__":
    train()
