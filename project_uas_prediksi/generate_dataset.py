import pandas as pd
import numpy as np
import random

# Set seed agar hasil konsisten
random.seed(42)
np.random.seed(42)

# Jumlah data
n = 100

# List jurusan
jurusan_choices = ["Akuntansi", "Manajemen", "Sistem Informasi", "Teknik Informatika"]

# Data simulasi
data = {
    "ipk": np.round(np.random.uniform(2.0, 4.0, n), 2),
    "sks": np.random.randint(110, 150, n),
    "kehadiran": np.random.randint(60, 100, n),
    "tidak_lulus": np.random.randint(0, 5, n),
    "organisasi": np.random.randint(0, 2, n),  # 0 = tidak aktif, 1 = aktif
    "semester": np.random.randint(6, 10, n),
    "jurusan": [random.choice(jurusan_choices) for _ in range(n)]
}

# Generate target label
target = []
for i in range(n):
    if data["ipk"][i] > 3.0 and data["kehadiran"][i] > 80 and data["tidak_lulus"][i] <= 1:
        target.append(1)
    else:
        target.append(0)

data["target"] = target

# Buat dataframe dan simpan
df = pd.DataFrame(data)
df.to_csv("dataset_kelulusan_realistic.csv", index=False)

print("✅ Dataset berhasil dibuat dan disimpan sebagai dataset_kelulusan_realistic.csv")
