import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import dagshub # <--- WAJIB IMPORT INI (NEW)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train():
    print("Mulai Training untuk Docker...")

    # --- BAGIAN PENTING (FIX) ---
    # Ini akan otomatis membaca Environment Variable (DAGSHUB_USERNAME & TOKEN)
    # dan mengkonfigurasi MLflow supaya BISA upload file ke DagsHub.
    # Ganti 'Eksperimen_SML_Alfarrel' dengan NAMA REPO DAGSHUB ANDA.
    dagshub.init(repo_owner='USERNAME_DAGSHUB_ANDA', repo_name='Eksperimen_SML_Alfarrel', mlflow=True)
    # ---------------------------
    
    # Cek apakah dataset ada
    if not os.path.exists("diabetes_clean.csv"):
        print("ERROR: File diabetes_clean.csv tidak ditemukan!")
        return

    # Load Data
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
        
        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        
        print("Sedang mengupload model ke DagsHub... (Tunggu sebentar)")
        
        # Log Model
        # Karena sudah ada dagshub.init(), ini sekarang akan SUKSES upload file
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Training Selesai. Run ID: {run.info.run_id}")
        
        # Simpan Run ID ke file txt
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

if __name__ == "__main__":
    train()