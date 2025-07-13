import os
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import pickle

# === 1. SETUP PATHS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "dataset_kelulusan_realistic.csv")
output_path = os.path.join(BASE_DIR, "selected_features.pkl")

# === 2. LOAD DATASET ===
df = pd.read_csv(dataset_path)

# Pastikan kolom 'jurusan' bertipe string
df["jurusan"] = df["jurusan"].astype(str)

# One-hot encoding kolom jurusan
df_encoded = pd.get_dummies(df, columns=["jurusan"])

# Pisahkan fitur dan target
X = df_encoded.drop(columns=["target"])
y = df_encoded["target"]

# Normalisasi (wajib untuk chi2)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Chi-Square selection
selector = SelectKBest(score_func=chi2, k=8)
selector.fit(X_scaled, y)

# Ambil fitur terpilih
mask = selector.get_support()
selected_features = X.columns[mask].tolist()

# Simpan ke file .pkl
with open(output_path, "wb") as f:
    pickle.dump(selected_features, f)

# Print info
print("✅ Feature selection selesai.")
print("🎯 Fitur terpilih:", selected_features)
