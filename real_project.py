import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st

# Memuat dataset
def muat_data(file):
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        st.error(f"Error memuat data: {e}")
        return None

# Persiapan data
def persiapkan_data(data):
    # Konversi 'Lokasi' menjadi one-hot encoding
    data = pd.get_dummies(data, columns=['Lokasi'], drop_first=True)
    X = data.drop(columns='Harga_Per_m2')  # Semua kolom kecuali target
    y = data['Harga_Per_m2']  # Target adalah harga per m2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test, data

# Melatih model
def latih_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Prediksi harga tanah
def prediksi_harga(model, inputs):
    try:
        return model.predict(inputs)
    except Exception as e:
        st.error(f"Error saat melakukan prediksi: {e}")
        return None

# Aplikasi utama
def main():
    st.title("Estimasi Harga Tanah Berdasarkan Lokasi, Luas, dan Harga per Meter")
    st.write("Aplikasi ini memprediksi harga tanah total berdasarkan lokasi, luas tanah, dan harga per meter persegi.")

    # Unggah file dataset
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file is not None:
        # Memuat dataset
        data = muat_data(uploaded_file)
        if data is None:
            st.error("Gagal memuat data.")
            return

        # Validasi dataset
        if not all(col in data.columns for col in ['Lokasi', 'Harga_Per_m2', 'Luas_Tanah']):
            st.error("Dataset harus memiliki kolom 'Lokasi', 'Harga_Per_m2', dan 'Luas_Tanah'.")
            return

        # Persiapan data
        X_train, X_test, y_train, y_test, data_ready = persiapkan_data(data)

        # Melatih model
        model = latih_model(X_train, y_train)

        # Form input untuk prediksi harga
        with st.form("form_input"):
            lokasi = st.selectbox("Pilih Lokasi", data['Lokasi'].unique())
            luas_tanah = st.number_input("Luas Tanah (m^2)", min_value=1, step=1, value=100)
            submitted = st.form_submit_button("Hitung Estimasi Harga")

        if submitted:
            # Siapkan input prediksi
            lokasi_dummies = pd.get_dummies([lokasi], prefix='Lokasi', drop_first=True).reindex(columns=data_ready.drop(columns='Harga_Per_m2').columns, fill_value=0)
            lokasi_dummies['Luas_Tanah'] = luas_tanah

            # Lakukan prediksi
            harga_per_meter = prediksi_harga(model, lokasi_dummies)
            if harga_per_meter is not None:
                total_harga = harga_per_meter[0] * luas_tanah
                st.success(f"Estimasi Harga Per Meter: Rp {harga_per_meter[0]:,.2f}")
                st.success(f"Estimasi Total Harga Tanah: Rp {total_harga:,.2f}")
    else:
        st.warning("Silakan unggah file CSV untuk melanjutkan.")

# Jalankan aplikasi
if __name__ == "__main__":
    main()
