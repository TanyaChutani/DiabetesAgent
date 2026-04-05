import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import requests

st.title("Diabetes Assistant")

query = st.text_input("Ask something")

if query:
    prompt = f"Answer clearly:\n{query}"

    res = requests.post(
        "http://localhost:8000/generate",
        json={"prompt": prompt}
    )

    st.write(res.json()["text"])