# Prediksi Harga Emas Menggunakan Model Time Series
Proyek ini bertujuan untuk mengembangkan dan mengevaluasi berbagai model *machine learning* untuk memprediksi harga emas di masa depan. Analisis ini menggunakan data historis harga emas dari tahun 2013 hingga 2023 dan membandingkan performa dari empat jenis model yang berbeda: **ARIMA, XGBoost, LSTM, dan GRU**.

## 📊 Alur Kerja Proyek
Notebook `PredectionWeb.ipynb` menyajikan alur kerja lengkap untuk proyek peramalan deret waktu (*time series forecasting*), yang mencakup langkah-langkah berikut:
1.  **Dataset**
      * Proyek ini menggunakan dataset "Gold Price (2013 - 2023)" yang diperoleh dari Kaggle.
      * Fitur utama yang digunakan untuk prediksi adalah kolom `Price`.

2.  **Pra-pemrosesan Data**
      * Data tanggal diubah menjadi format *datetime* dan ditetapkan sebagai indeks DataFrame.
      * **Penskalaan Fitur**: Untuk meningkatkan konvergensi dan performa model (terutama untuk model berbasis *neural network*), data harga diskalakan ke rentang antara 0
      * dan 1 menggunakan `MinMaxScaler` dari Scikit-learn.

3.  **Pelatihan dan Evaluasi Model**
      * Empat model yang berbeda dilatih dan dievaluasi untuk menemukan yang paling akurat dalam memprediksi harga emas:
        1.  **ARIMA**: Model statistik klasik yang populer untuk analisis deret waktu.
        2.  **XGBoost**: Model *gradient boosting* yang dikenal dengan kecepatan dan performanya yang tinggi.
        3.  **LSTM (Long Short-Term Memory)**: Model *deep learning* dari jenis *Recurrent Neural Network* (RNN) yang sangat efektif untuk data sekuensial.
        4.  **GRU (Gated Recurrent Unit)**: Varian dari RNN yang mirip dengan LSTM tetapi dengan arsitektur yang lebih sederhana.
      * Setiap model dievaluasi kinerjanya menggunakan metrik **Mean Squared Error (MSE)** dan **R2 Score**.
      * Visualisasi perbandingan antara harga aktual dan hasil prediksi dibuat untuk setiap model guna menganalisis performa secara kualitatif.

4.  **Hasil dan Kesimpulan**
      * Berdasarkan alur kerja dalam *notebook*, model **GRU** dipilih sebagai model dengan performa terbaik.
      * Model ini menunjukkan kemampuan yang baik dalam menangkap pola dan tren dari data deret waktu harga emas.

## 🚀 Penggunaan dan Implementasi
Untuk memungkinkan penggunaan di masa depan atau untuk diintegrasikan ke dalam aplikasi, komponen-komponen penting dari proyek ini telah disimpan:
  * **Model Terbaik**: Model GRU yang telah dilatih disimpan dalam file `model_gru.h5`.
  * **Scaler**: Objek `MinMaxScaler` yang digunakan untuk data fitur (`scaler_X.save`) dan data target (`scaler_y.save`) juga disimpan. Ini penting agar data baru dapat diproses dengan cara yang sama sebelum dimasukkan ke dalam model.

### Cara Menjalankan
1.  **Instalasi Dependensi**: Pastikan semua *library* yang dibutuhkan telah terpasang.
    ```bash
    pip install pandas numpy tensorflow scikit-learn statsmodels xgboost matplotlib seaborn joblib
    ```
2.  **Siapkan Dataset**: Unduh dataset `gold_price_data.csv` dan letakkan dalam direktori yang sama dengan *notebook*.
3.  **Jalankan Notebook**: Eksekusi semua sel di dalam `PredectionWeb.ipynb` untuk melatih, mengevaluasi, dan menyimpan model serta *scaler*.
