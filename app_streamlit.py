import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
	page_title = "Classification Level Coding"
)

model = joblib.load("model_klasifikasi_level_coding_siswa.joblib")

st.title("Classification Level Coding")
st.markdown("klasifikasi level coding berdasarkan fitur **hours_coding_daily**, **preferred_language**, **typing_speed**, **import_usage**, **oop_usage**")

hours_coding_daily = st.slider("Hours Coding Daily", 1.0, 5.5, 3.0)
preferred_language = st.pills("Preferred Language", ["Python", "C++", "Java"], default="Python")
typing_speed = st.slider("Typing Speed", 20, 65, 40)
import_usage = st.pills("Import Usage", ["Yes", "No"], default="No")
oop_usage = st.pills("OOP Usage", ["Yes", "No"], default="Yes")

if st.button("Predict", type="primary"):
	data_baru = pd.DataFrame([[hours_coding_daily, preferred_language, typing_speed, import_usage, oop_usage]], 
	columns=["hours_coding_daily", "preferred_language", "typing_speed", "import_usage", "oop_usage"])
	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f"model memprediksi {prediksi} dengan tingkat keyakinan {presentase*100:.2f}%")
	st.balloons()

st.divider()
st.caption("Dibuat oleh Nadya Nurjzillani")