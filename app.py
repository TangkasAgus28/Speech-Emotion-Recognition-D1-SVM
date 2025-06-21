import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
from io import BytesIO

# ==========================
# Konfigurasi Halaman Streamlit
# ==========================
st.set_page_config(
    page_title="Speech Emotion Recognition",
    layout="wide",
    initial_sidebar_state="collapsed", # Bisa "expanded" atau "collapsed"
    page_icon="🎤"
)

# ==========================
# Load Model dan Komponen 
# ==========================
try:
    best_model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    st.error("File model (best_model.pkl, scaler.pkl, label_encoder.pkl) tidak ditemukan. Pastikan mereka berada di direktori yang sama dengan aplikasi Streamlit.")
    st.stop() 

# ==========================
# Fungsi Ekstraksi Fitur
# ==========================
def extract_features(y, sr=22050):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y=y).mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    # Pastikan outputnya 1D array
    return np.concatenate([mfcc, [zcr, rms, centroid]])

# ==========================
# Kernel dan Prediksi Model (Seperti kode Anda)
# ==========================
def rbf_kernel(X1, X2, gamma):
    X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
    dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * dist_sq)

def predict_svm_rbf(X_train, X_test, alpha, support_idx, b, support_vectors, support_labels, gamma):
    K = rbf_kernel(X_test, support_vectors, gamma)
    return np.sign(np.dot(K, alpha[support_idx] * support_labels) + b)

def predict_ovr(models, X):
    votes = []

    for cls, alpha, support_idx, b, sv, sl, gamma in models:
       
        y_pred = predict_svm_rbf(sv, X, alpha, support_idx, b, sv, sl, gamma)
        votes.append(y_pred)
    votes = np.array(votes)

    return np.argmax(votes, axis=0) 


# ==========================
# Custom CSS 
# ==========================
st.markdown(
    """
    <style>
    /* Mengubah warna teks utama untuk kontras yang lebih baik pada tema gelap */
    body {
        color: #e0e0e0;
    }
    .stApp {
        background-color: #0c0c0c; /* Latar belakang gelap */
    }
    /* Mempercantik kotak upload */
    .stFileUploader {
        border: 2px dashed #4CAF50; /* Warna hijau menarik */
        padding: 20px;
        border-radius: 10px;
        background-color: #1a1a1a;
        margin-bottom: 20px;
    }
    .stFileUploader > div > div > div:nth-child(2) { /* Tombol browse */
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 8px 15px;
    }
    /* Style untuk judul dan subjudul */
    h1 {
        color: #4CAF50;
        text-align: center;
        font-size: 3em;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 5px #000000;
    }
    h4 {
        color: #a0a0a0;
        text-align: center;
        margin-top: 0;
        margin-bottom: 2em;
    }
    .stAudio { /* Menambah sedikit margin bawah pada audio player */
        margin-bottom: 20px;
    }
    /* Style untuk kotak hasil prediksi */
    .stSuccess {
        background-color: #28a745; /* Hijau sukses */
        color: white;
        border-radius: 10px;
        padding: 15px;
        font-size: 1.5em;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 30px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stInfo { /* Jika ingin pakai st.info untuk pesan */
        background-color: #17a2b8;
        color: white;
        border-radius: 10px;
        padding: 15px;
        font-size: 1.2em;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 30px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* Memastikan plot Matplotlib terlihat bagus di tema gelap */
    .matplotlib {
        background-color: #1a1a1a !important;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================
# Antarmuka Utama Streamlit
# ==========================

# Hero Section
st.header("🎤 Speech Emotion Recognition", divider='green')
st.markdown(
    """
    <p style="text-align: left; font-size: 1.1em; color: #b0b0b0;">
    Unggah file audio (.wav) Anda di sini untuk menganalisis emosi yang terkandung di dalamnya.
    Sistem akan memproses sinyal suara menggunakan <b>MFCC, Zero Crossing Rate, RMS, dan Spectral Centroid</b>
    untuk memprediksi emosi seperti senang, sedih, marah, netral, dan lainnya.
    </p>
    """,
    unsafe_allow_html=True
)

st.write("---") # Garis pemisah

# Upload Section
st.markdown(
    """
    <h3 style="color: #4CAF50;">Upload Audio File</h3>
    <p style="color: #a0a0a0;">Format yang didukung: <b>.wav</b> (maks. 20MB)</p>
    """,
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader(" ", type=["wav"], accept_multiple_files=False, key="audio_uploader") # Label kosong agar ikon lebih menonjol


if uploaded_file is not None:
    st.info("Memproses audio Anda... Mohon tunggu sebentar.")
    

    with st.spinner('Menganalisis sinyal suara...'):
        try:
            # Baca audio
            y, sr = librosa.load(uploaded_file, sr=22050)
            y = librosa.util.normalize(y)
            y, _ = librosa.effects.trim(y) # Hapus silent part

        
            duration = 3.0 
            desired_len = int(sr * duration)
            if len(y) > desired_len:
                y = y[:desired_len] # Potong jika terlalu panjang
            elif len(y) < desired_len:
                y = np.pad(y, (0, desired_len - len(y)), 'constant') # Pad jika terlalu pendek

            # Ekstraksi fitur dan prediksi
            feature = extract_features(y, sr=sr).reshape(1, -1)
            feature_scaled = scaler.transform(feature)
            
            # Melakukan prediksi
            pred_enc = predict_ovr(best_model, feature_scaled)
       
            if isinstance(pred_enc, np.ndarray) and pred_enc.size > 0:
                pred_label = label_encoder.inverse_transform([int(pred_enc.item())])[0]
            else:
                st.error("Gagal melakukan prediksi. Hasil prediksi tidak valid.")
                pred_label = "Tidak Diketahui"

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file audio: {e}")
            st.stop() 


    # Tampilkan Audio dan Hasil Prediksi
    st.markdown("---") # Garis pemisah
    st.subheader("🎵 Audio Anda")
    st.audio(uploaded_file, format='audio/wav')

    # Tampilan hasil prediksi yang lebih menonjol
    st.markdown(
    f"""
    <div class="stSuccess">
        <h3>🎉 Emosi Terdeteksi: <span style='color: white; font-size: 1.8em;'>{pred_label.upper()}</span></h3>
    </div>
    """,
    unsafe_allow_html=True
    )

    st.markdown("---") # Garis pemisah
    st.subheader("🔬 Analisis Sinyal (Domain Waktu, Frekuensi & Waktu-Frekuensi)")

    col1, col2, col3 = st.columns(3)

    # ======================
    # Waveform
    # ======================
    with col1:
        st.markdown("<h5 style='text-align: center; color: #e0e0e0;'>🕒 Waveform (Domain Waktu)</h5>", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax1, color='#6495ED') # Warna biru
        ax1.set_title("Representasi Domain Waktu", color='white')
        ax1.set_xlabel("Waktu (s)", color='white')
        ax1.set_ylabel("Amplitudo", color='white')
        ax1.tick_params(colors='white') # Warna tick labels
        ax1.set_facecolor('#1a1a1a') # Warna latar belakang plot
        fig1.set_facecolor('#1a1a1a') # Warna latar belakang figure
        st.pyplot(fig1)
        plt.close(fig1) 

    # ======================
    # FFT / Frekuensi
    # ======================
    with col2:
        st.markdown("<h5 style='text-align: center; color: #e0e0e0;'>🔊 Spektrum Frekuensi (FFT)</h5>", unsafe_allow_html=True)
        fft = np.fft.fft(y)
        magnitude = np.abs(fft)
        frequency = np.linspace(0, sr, len(magnitude))
        half_range = len(magnitude) // 2
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(frequency[:half_range], magnitude[:half_range], color='#FFD700') # Warna emas
        ax2.set_title("Representasi Domain Frekuensi", color='white')
        ax2.set_xlabel("Frekuensi (Hz)", color='white')
        ax2.set_ylabel("Magnitudo", color='white')
        ax2.tick_params(colors='white')
        ax2.set_facecolor('#1a1a1a')
        fig2.set_facecolor('#1a1a1a')
        st.pyplot(fig2)
        plt.close(fig2)

    # ======================
    # MFCC
    # ======================
    with col3:
        st.markdown("<h5 style='text-align: center; color: #e0e0e0;'>📊 Mel-frequency Cepstral Coefficients (MFCC)</h5>", unsafe_allow_html=True)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax3, cmap='viridis') # Warna colormap
        fig3.colorbar(img, ax=ax3, format="%+2.f").ax.yaxis.set_tick_params(color='white')
        ax3.set_title("Representasi Waktu-Frekuensi (MFCC)", color='white')
        ax3.set_xlabel("Waktu (s)", color='white')
        ax3.set_ylabel("MFCC Koefisien", color='white')
        ax3.tick_params(colors='white')
        ax3.set_facecolor('#1a1a1a')
        fig3.set_facecolor('#1a1a1a')
        st.pyplot(fig3)
        plt.close(fig3)

# Footer
st.markdown("---")
st.markdown(
    """
    <p style="text-align: center; color: #707070; font-size: 0.9em;">
    Oleh D_1 > Speech Emotion Recognition.
    </p>
    """,
    unsafe_allow_html=True
)