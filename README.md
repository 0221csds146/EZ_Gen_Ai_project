# 📚 EZ GenAI – Smart Assistant for Research Summarization

An AI-powered research assistant built with **Streamlit**, **LangChain**, and **Groq API**, designed to intelligently **summarize**, **query**, and **challenge users** with MCQs based on uploaded PDF/TXT research documents.

🔗 **Live App**: [https://ezgenaiproject-rajeevkumar.streamlit.app/](https://ezgenaiproject-rajeevkumar.streamlit.app/)

---

## 🚀 Features

- ✅ Upload and process `.pdf` or `.txt` documents
- ✨ AI-generated **summary**
- 🔍 Ask **custom questions** with source references
- 🧠 "Challenge Me" mode with **logic-based MCQs** and answer feedback
- 💬 **Conversational Q&A** using memory buffer
- ⚡ Fast response powered by **Groq’s LLaMA3 LLMs**

---

## 🧱 Architecture & Reasoning Flow

```mermaid
flowchart TD
    A[User Uploads PDF/TXT] --> B[Text Extraction]
    B --> C[Text Chunking & Embedding via FAISS]
    C --> D[Summarization with LLM]
    C --> E[Q&A with RetrievalQA]
    C --> F[Logic-Based MCQ Generation]
    F --> G[User Response Evaluation]
    E --> H[Answer + Source Snippets]
    G --> I[Feedback on Answers]
