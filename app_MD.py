import streamlit as st
import pandas as pd
import joblib

# 1. KONFIGURASI HALAMAN (Layout Wide agar lebih luas)
st.set_page_config(page_title="Prediksi Karir Mahasiswa", layout="wide")

# 2. SIDEBAR 

st.sidebar.title("Tentang Aplikasi")
st.sidebar.info("App prediksi memakai model Random Forest untuk menganalisis potensi penempatan kerja dan estimasi gaji awal.")
st.sidebar.markdown("---")
st.sidebar.write("UTS Model Deployment")
st.sidebar.write("Miryam Almira Levina - 2802465944")

# Header Utama
st.title("Dashboard Prediksi Karir & Gaji Mahasiswa")
st.markdown("Masukkan data akademik dan portofolio Anda pada formulir di bawah ini.")

# 3. LOAD MODEL
@st.cache_resource
def load_models():
    model_cls = joblib.load('model_klasifikasi.pkl')
    model_reg = joblib.load('model_regresi.pkl')
    return model_cls, model_reg

try:
    model_klasifikasi, model_regresi = load_models()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# 4. FORMULIR INPUT
with st.form("form_prediksi"):
    st.subheader("Formulir Profil Mahasiswa")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Riwayat Akademik**")
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        cgpa = st.number_input("CGPA (Skala 10)", 0.0, 10.0, 7.5, 0.1)
        ssc_percentage = st.slider("Nilai SMA (SSC %)", 0, 100, 70)
        hsc_percentage = st.slider("Nilai K-12 (HSC %)", 0, 100, 70)
        degree_percentage = st.slider("Nilai Sarjana (%)", 0, 100, 75)
        
    with col2:
        st.markdown("**Keahlian & Skor**")
        entrance_exam_score = st.slider("Skor Ujian Masuk", 0, 100, 70)
        technical_skill_score = st.slider("Skor Keahlian Teknis", 0, 100, 80)
        soft_skill_score = st.slider("Skor Soft Skill", 0, 100, 80)
        attendance_percentage = st.slider("Kehadiran (%)", 0, 100, 85)
        backlogs = st.number_input("Jumlah Mengulang (Backlog)", 0, 10, 0)

    with col3:
        st.markdown("**Pengalaman & Portofolio**")
        work_experience_months = st.number_input("Pengalaman Kerja (Bulan)", 0, 60, 0)
        internship_count = st.number_input("Jumlah Magang", 0, 10, 1)
        live_projects = st.number_input("Jumlah Proyek Nyata", 0, 20, 2)
        certifications = st.number_input("Jumlah Sertifikasi", 0, 10, 1)
        extracurricular_activities = st.selectbox("Ikut Ekstrakurikuler?", ["Yes", "No"])

    # Tombol submit di dalam form
    st.markdown("---")
    submitted = st.form_submit_button("Prediksi Hasil Karir", use_container_width=True)

# 5. LOGIKA PREDIKSI & DATA VISUALIZATION
if submitted:
    # Menyusun data input
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
    st.header("Laporan Hasil Prediksi")
    
    # Layout hasil 
    res_col1, res_col2 = st.columns([1, 1])
    
    # Prediksi
    pred_placement = model_klasifikasi.predict(input_data)[0]
    
    with res_col1:
        if pred_placement == 1:
            st.success("Mahasiswa diprediksi DITERIMA (Placed).**")
            pred_salary = model_regresi.predict(input_data)[0]
            st.metric(label="Estimasi Penawaran Gaji (LPA)", value=f"{pred_salary:.2f} Lakhs", delta="Sangat Kompetitif")
        else:
            st.error(" **Mahasiswa diprediksi BELUM DITERIMA (Not Placed).**")
            st.metric(label="Estimasi Penawaran Gaji", value="0.00 Lakhs", delta="- Perlu Peningkatan", delta_color="inverse")
            st.warning("Saran: Tingkatkan skor teknikal atau tambah portofolio proyek/magang Anda.")

    # Data Visualization 
    with res_col2:
        st.markdown("** Visualisasi Profil Keahlian Mahasiswa**")
        
        df_viz = pd.DataFrame({
            "Kategori": ["Keahlian Teknis", "Soft Skill", "Ujian Masuk", "Nilai Sarjana"],
            "Skor": [technical_skill_score, soft_skill_score, entrance_exam_score, degree_percentage]
        }).set_index("Kategori")
        
        #  Bar Chart
        st.bar_chart(df_viz, use_container_width=True)