import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- KONFIGURASI DAGSHUB (Versi Aman / CI-CD Friendly) ---
# Kita ambil value dari GitHub Secrets yang dikirim lewat Environment Variable
dagshub_username = os.environ.get("MLFLOW_TRACKING_USERNAME")
dagshub_token = os.environ.get("MLFLOW_TRACKING_PASSWORD")

if not dagshub_username or not dagshub_token:
    print("⚠️ WARNING: Credential DagsHub tidak terdeteksi. Pastikan GitHub Secrets sudah diset!")

# Set URI secara manual (karena ini cuma link, tidak rahasia)
# Ganti 'Eksperimen_SML_Alfarrel' dengan NAMA REPO DAGSHUB KAMU yang benar
# Format: https://dagshub.com/<USERNAME>/<NAMA_REPO>.mlflow
mlflow.set_tracking_uri(f"https://dagshub.com/{dagshub_username}/Eksperimen_SML_Alfarrel.mlflow")

# Setup auth biar MLflow bisa login
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

def train():
    print("Mulai Training untuk Docker...")
    
    # Load Data (Pastikan file csv ada di folder yang sama)
    df = pd.read_csv("diabetes_clean.csv")
    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Model Setup
    model = RandomForestClassifier(n_estimators=50)
    
    # Start Run
    mlflow.set_experiment("Eksperimen_CI_Docker")
    
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        
        # Log Metrics & Model
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Training Selesai. Run ID: {run.info.run_id}")
        
        # --- BAGIAN PENTING BUAT DOCKER ---
        # Kita simpan ID run ini ke file txt biar bisa dibaca GitHub Actions
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

if __name__ == "__main__":
    train()