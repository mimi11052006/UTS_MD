import streamlit as st
import pandas as pd
import joblib

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Karir Mahasiswa", page_icon="🎓", layout="centered")
st.title("🎓 Sistem Prediksi Penempatan Kerja & Gaji")
st.write("Aplikasi ini memprediksi apakah mahasiswa akan mendapatkan pekerjaan dan estimasi gajinya berdasarkan profil akademik dan keahlian.")

# 2. Load Model (Gunakan cache agar model tidak dimuat ulang setiap kali ada interaksi)
@st.cache_resource
def load_models():
    # Pastikan nama file sesuai dengan yang Anda buat di Nomor 2
    model_cls = joblib.load('model_klasifikasi.pkl')
    model_reg = joblib.load('model_regresi.pkl')
    return model_cls, model_reg

try:
    model_klasifikasi, model_regresi = load_models()
    st.sidebar.success("✅ Model berhasil dimuat!")
except Exception as e:
    st.error(f"⚠️ Gagal memuat model: {e}")
    st.stop()

# 3. Form Input Data Mahasiswa
st.header("📝 Masukkan Data Mahasiswa")

# Membagi layar menjadi 2 kolom agar rapi
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    cgpa = st.number_input("CGPA (Skala 10)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    ssc_percentage = st.number_input("Nilai SMA (SSC %)", min_value=0, max_value=100, value=70)
    hsc_percentage = st.number_input("Nilai K-12 (HSC %)", min_value=0, max_value=100, value=70)
    degree_percentage = st.number_input("Nilai Sarjana (%)", min_value=0, max_value=100, value=70)
    entrance_exam_score = st.number_input("Skor Ujian Masuk", min_value=0, max_value=100, value=70)
    attendance_percentage = st.number_input("Kehadiran (%)", min_value=0, max_value=100, value=80)
    backlogs = st.number_input("Jumlah Mata Kuliah Mengulang (Backlogs)", min_value=0, max_value=10, value=0)

with col2:
    technical_skill_score = st.number_input("Skor Keahlian Teknis", min_value=0, max_value=100, value=75)
    soft_skill_score = st.number_input("Skor Soft Skill", min_value=0, max_value=100, value=80)
    work_experience_months = st.number_input("Pengalaman Kerja (Bulan)", min_value=0, max_value=60, value=0)
    internship_count = st.number_input("Jumlah Magang", min_value=0, max_value=10, value=1)
    live_projects = st.number_input("Jumlah Proyek Nyata", min_value=0, max_value=20, value=2)
    certifications = st.number_input("Jumlah Sertifikasi", min_value=0, max_value=10, value=1)
    extracurricular_activities = st.selectbox("Kegiatan Ekstrakurikuler", ["Yes", "No"])

# 4. Tombol Prediksi
if st.button("🚀 Prediksi Hasil Karir", use_container_width=True):
    # Menyusun data ke dalam DataFrame (Nama kolom WAJIB sama persis dengan dataset X_train)
    input_data = pd.DataFrame({
        'gender': [gender],
        'ssc_percentage': [ssc_percentage],
        'hsc_percentage': [hsc_percentage],
        'degree_percentage': [degree_percentage],
        'cgpa': [cgpa],
        'entrance_exam_score': [entrance_exam_score],
        'technical_skill_score': [technical_skill_score],
        'soft_skill_score': [soft_skill_score],
        'internship_count': [internship_count],
        'live_projects': [live_projects],
        'work_experience_months': [work_experience_months],
        'certifications': [certifications],
        'attendance_percentage': [attendance_percentage],
        'backlogs': [backlogs],
        'extracurricular_activities': [extracurricular_activities]
    })
    
    st.markdown("---")
    st.header("🎯 Hasil Prediksi")
    
    # Menjalankan Prediksi Klasifikasi
    pred_placement = model_klasifikasi.predict(input_data)[0]
    
    if pred_placement == 1:
        st.success("🎉 **Status Penempatan: DITERIMA (Placed)**")
        
        # Jika diterima, jalankan Prediksi Regresi untuk Gaji
        pred_salary = model_regresi.predict(input_data)[0]
        st.info(f"💰 **Estimasi Gaji (LPA): {pred_salary:.2f} Lakhs Per Annum**")
    else:
        st.error("📉 **Status Penempatan: BELUM DITERIMA (Not Placed)**")
        st.warning("💡 Tingkatkan pengalaman kerja, CGPA, atau skor teknis Anda untuk memperbesar peluang!")