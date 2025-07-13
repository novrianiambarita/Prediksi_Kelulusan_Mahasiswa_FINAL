import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# === 1. SETUP PATHS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "dataset_kelulusan_realistic.csv")
features_path = os.path.join(BASE_DIR, "selected_features.pkl")
model_path = os.path.join(BASE_DIR, "logistic_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

# === 2. LOAD DATASET & FITUR ===
df = pd.read_csv(dataset_path)

# Pastikan 'jurusan' adalah kolom string
df["jurusan"] = df["jurusan"].astype(str)

# One-hot encoding untuk jurusan
df_encoded = pd.get_dummies(df, columns=["jurusan"])

# Pisahkan fitur dan target
X = df_encoded.drop(columns=["target"])
y = df_encoded["target"]

# Simpan nama fitur yang dipakai
selected_features = X.columns.tolist()

# === 3. STANDARISASI ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === 5. TRAIN MODEL ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# === 6. SAVE MODEL, SCALER, FITUR ===
with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

with open(features_path, "wb") as f:
    pickle.dump(selected_features, f)

# === 7. SUCCESS MESSAGE ===
print("✅ Model training selesai.")
print("📦 Model disimpan:", model_path)
print("📦 Scaler disimpan:", scaler_path)
print("📦 Fitur disimpan:", features_path)
