import streamlit as st
import requests
import json

backend_url = "http://127.0.0.1:8000"

def set_api_key(api_key):
    url = f"{backend_url}/set_api_key"
    data = {"api_key": api_key}
    response = requests.post(url, json=data)
    return response.json()

def upload_file(file):
    url = f"{backend_url}/upload"
    files = {"file": (file.name, file.getvalue())}
    response = requests.post(url, files=files)
    return response.json()

def ask_question(question):
    url = f"{backend_url}/ask"
    data = {"question": question}
    response = requests.post(url, json=data)
    return response.json()

st.title("Document Question Answering System")

# Step 1: Set API Key
st.sidebar.header("Step 1: Set API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if st.sidebar.button("Set API Key"):
    if api_key:
        result = set_api_key(api_key)
        st.sidebar.success(result.get("message", "API key set successfully"))
    else:
        st.sidebar.error("API key cannot be empty")

# Step 2: Upload PDF
st.sidebar.header("Step 2: Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
if st.sidebar.button("Upload PDF"):
    if uploaded_file:
        result = upload_file(uploaded_file)
        st.sidebar.success(result.get("message", "File uploaded successfully"))
    else:
        st.sidebar.error("Please select a PDF file to upload")

# Step 3: Ask Questions
st.header("Ask Questions about the Uploaded Document")
question = st.text_input("Enter your question")
if st.button("Ask Question"):
    if question:
        result = ask_question(question)
        st.write(result.get("answer", "No answer available"))
    else:
        st.error("Question cannot be empty")
