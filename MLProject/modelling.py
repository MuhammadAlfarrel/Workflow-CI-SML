import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train():
    print("=== Training MLflow Project ===")

    df = pd.read_csv("diabetes_clean.csv")
    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    # Log metric
    mlflow.log_metric("accuracy", acc)

    # Log model → artifact_path WAJIB "model"
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    print("Training selesai")

if __name__ == "__main__":
    train()
