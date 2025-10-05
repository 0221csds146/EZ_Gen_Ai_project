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

# Reusable function to get the LLM with a supported model
def get_groq_llm(model="groq-llama2-13b", temperature=0.0):
    """
    Returns a ChatGroq instance using a supported Groq model.
    Parameters:
        model (str): Name of the supported Groq model.
        temperature (float): Controls creativity/randomness of output.
    """
    return ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model,
        temperature=temperature
    )
