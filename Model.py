
import os
import streamlit as st
import gdown

FILE_ID   = "your_file_id_here"
DEST_PATH = "model.pkl"
URL       = f"https://drive.google.com/file/d/1ucW1zGVKxPWM0BlpOdnBUEvXizWHDVuK/view?usp=drive_link"

if not os.path.exists(DEST_PATH):
    with st.spinner("Fetching model from Google Drive…"):
        gdown.download(URL, DEST_PATH, quiet=False)

# load it
import pickle
model = pickle.load(open(DEST_PATH, "rb"))
