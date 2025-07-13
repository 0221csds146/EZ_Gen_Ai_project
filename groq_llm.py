import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Set up the Groq API Key
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Reusable function to get the LLM
def get_groq_llm(model="llama3-8b-8192", temperature=0.0):
    return ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model,
        temperature=temperature
    )
