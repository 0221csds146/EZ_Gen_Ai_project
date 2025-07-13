import streamlit as st
from utils import extract_text_from_pdf, extract_text_from_txt
from backend import (
    summarize_document,
    prepare_vector_store,
    qa_chain,
    generate_logic_questions,
    evaluate_user_response,
    get_conversational_chain
)

st.set_page_config(page_title="📘 Smart Research Assistant", layout="wide")
st.title("📘 Smart Assistant for Research Summarization")

# File Upload
uploaded_file = st.file_uploader("Upload your PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    file_text = extract_text_from_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else extract_text_from_txt(uploaded_file)

    st.subheader("📄 Auto Summary")
    summary = summarize_document(file_text)
    st.success(summary)

    # Vector Store and Conversational Chain
    vector_store = prepare_vector_store(file_text)

    if "chat_chain" not in st.session_state:
        st.session_state.chat_chain = get_conversational_chain(vector_store)

    mode = st.radio("Choose Mode", ["Ask Anything", "Challenge Me"])

    if mode == "Ask Anything":
        user_question = st.text_input("Ask a question (you can follow up):")
        if user_question:
            response = st.session_state.chat_chain.run(user_question)
            st.markdown("### 📌 Answer")
            st.write(response)

    elif mode == "Challenge Me":
        if "challenge_questions" not in st.session_state:
            st.session_state.challenge_questions = []
            st.session_state.challenge_history = []

        if st.button("🎯 Generate Questions"):
            questions_block = generate_logic_questions(file_text)
            st.session_state.challenge_questions = [q.strip("-•123. ") for q in questions_block.split("\n") if q.strip()]

        if st.session_state.challenge_questions:
            st.markdown("### 💡 Answer These Questions")
            for idx, question in enumerate(st.session_state.challenge_questions):
                user_answer = st.text_input(f"Q{idx + 1}: {question}", key=f"challenge_q_{idx}")
                if user_answer:
                    feedback = evaluate_user_response(file_text, question, user_answer)
                    st.session_state.challenge_history.append({
                        "question": question,
                        "user_response": user_answer,
                        "evaluation": feedback
                    })

        if st.session_state.challenge_history:
            st.markdown("### ✅ Feedback Summary")
            for i, entry in enumerate(st.session_state.challenge_history):
                st.markdown(f"**Q{i+1}:** {entry['question']}")
                st.markdown(f"**Your Answer:** {entry['user_response']}")
                st.markdown(f"**Feedback:** {entry['evaluation']}")
                st.markdown("---")
