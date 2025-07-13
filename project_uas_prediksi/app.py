from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import pickle
import pandas as pd
from datetime import datetime

# === Path dan Direktori ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = BASE_DIR

# === Inisialisasi Flask ===
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# === Load Model, Scaler, dan Fitur ===
model = pickle.load(open(os.path.join(BASE_DIR, 'logistic_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb'))
selected_features = pickle.load(open(os.path.join(BASE_DIR, 'selected_features.pkl'), 'rb'))

# === Penyimpanan Riwayat ===
riwayat_data = []

# === Halaman Utama ===
@app.route('/')
def index():
    return render_template("index.html", riwayat=riwayat_data, result=None, edit_data=None, edit_index=None)

# === Prediksi Baru ===
@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    input_data = {
        "ipk": float(form["ipk"]),
        "sks": int(form["sks"]),
        "kehadiran": int(form["kehadiran"]),
        "tidak_lulus": int(form["tidak_lulus"]),
        "organisasi": int(form["organisasi"]),
        "semester": int(form["semester"]),
        "jurusan_Akuntansi": 0,
        "jurusan_Manajemen": 0,
        "jurusan_Sistem Informasi": 0,
        "jurusan_Teknik Informatika": 0,
    }

    jurusan_key = "jurusan_" + form["jurusan"]
    if jurusan_key in input_data:
        input_data[jurusan_key] = 1

    df = pd.DataFrame([input_data])
    X = df[selected_features]
    X_scaled = scaler.transform(X)
    hasil = model.predict(X_scaled)[0]
    hasil_text = "Lulus" if hasil == 1 else "Tidak Lulus"

    # Simpan ke riwayat
    data_riwayat = input_data.copy()
    data_riwayat["jurusan"] = form["jurusan"]
    data_riwayat["hasil"] = hasil_text
    data_riwayat["waktu"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    riwayat_data.append(data_riwayat)

    return render_template("index.html", result=hasil_text, riwayat=riwayat_data, edit_data=None, edit_index=None)

# === Edit Data ===
@app.route('/edit/<int:index>', methods=['GET', 'POST'])
def edit(index):
    if request.method == 'POST':
        form = request.form
        updated_data = {
            "ipk": float(form["ipk"]),
            "sks": int(form["sks"]),
            "kehadiran": int(form["kehadiran"]),
            "tidak_lulus": int(form["tidak_lulus"]),
            "organisasi": int(form["organisasi"]),
            "semester": int(form["semester"]),
            "jurusan": form["jurusan"],
            "waktu": riwayat_data[index]["waktu"]
        }

        # One-hot encoding ulang
        for jur in ["Akuntansi", "Manajemen", "Sistem Informasi", "Teknik Informatika"]:
            updated_data[f"jurusan_{jur}"] = 1 if form["jurusan"] == jur else 0

        df = pd.DataFrame([updated_data])
        X = df[selected_features]
        X_scaled = scaler.transform(X)
        hasil = model.predict(X_scaled)[0]
        updated_data["hasil"] = "Lulus" if hasil == 1 else "Tidak Lulus"

        riwayat_data[index] = updated_data
        return redirect(url_for("index"))
    else:
        edit_data = riwayat_data[index]
        return render_template("index.html", edit_data=edit_data, edit_index=index, riwayat=riwayat_data, result=None)

# === Hapus Data ===
@app.route('/delete/<int:index>')
def delete(index):
    if 0 <= index < len(riwayat_data):
        riwayat_data.pop(index)
    return redirect(url_for("index"))

# === Unduh CSV ===
@app.route('/download')
def download_csv():
    if not riwayat_data:
        return redirect(url_for("index"))

    df = pd.DataFrame(riwayat_data)
    csv_path = os.path.join(BASE_DIR, "riwayat_prediksi.csv")
    df.to_csv(csv_path, index=False)
    return send_file(csv_path, as_attachment=True)

# === Jalankan Aplikasi Flask ===
if __name__ == '__main__':
    print(f"📁 Template folder: {TEMPLATE_DIR}")
    app.run(debug=False, use_reloader=False)
