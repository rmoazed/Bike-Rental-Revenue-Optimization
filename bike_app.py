import streamlit as st
from pathlib import Path
import os

st.set_page_config(page_title="Test App", layout="wide")

st.title("Streamlit Deploy Test")

st.write("If you can see this, the app is working!")

st.write("Current working directory:", os.getcwd())
st.write("Files in repo root:", os.listdir("."))

models_path = Path("models")
st.write("Models folder exists:", models_path.exists())

if models_path.exists():
    st.write("Files in models folder:", os.listdir(models_path))
