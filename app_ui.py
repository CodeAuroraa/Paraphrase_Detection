#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import requests

st.title("Semantic Similarity & Paraphrase Detection")

text1 = st.text_area("Enter first sentence", "")
text2 = st.text_area("Enter second sentence", "")

if st.button("Check Similarity"):
    if text1 and text2:
        url = "https://paraphrase-api.onrender.com/paraphrase"  # Replace with your API URL
        response = requests.post(url, json={"text1": text1, "text2": text2})
        result = response.json()

        st.success(f"Cosine Similarity: {result['cosine_similarity']}")
        st.info(f"Paraphrase Detection: {result['paraphrase']}")
    else:
        st.error("Please enter both sentences.")

