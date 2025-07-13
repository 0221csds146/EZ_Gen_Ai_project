import streamlit as st
import os
from backend import (
    prepare_vector_store,
    summarize_document,
    qa_chain,
    generate_logic_questions,
    evaluate_user_response,
    get_conversational_chain
)

# Secure API key loading
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    except ImportError:
        st.error("Please install python-dotenv or set GROQ_API_KEY in Streamlit secrets")

# File reading logic with error handling
def read_file(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            import fitz  # PyMuPDF
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page in pdf_document:
                text += page.get_text() + "\n"
            pdf_document.close()
            return text
        elif uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        else:
            st.error("Unsupported file format. Please upload PDF or TXT files only.")
            return ""
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return ""

# Streamlit UI
st.set_page_config(page_title="Smart Research Assistant", layout="wide")
st.title("📚 Smart Assistant for Research Summarization")

# Initialize session state variables
if "logic_questions" not in st.session_state:
    st.session_state.logic_questions = None
if "questions_loaded" not in st.session_state:
    st.session_state.questions_loaded = False
if "current_document" not in st.session_state:
    st.session_state.current_document = None
if "question_generation_attempts" not in st.session_state:
    st.session_state.question_generation_attempts = 0

# File upload
uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])

if uploaded_file:
    # Read file content
    with st.spinner("Reading document..."):
        file_text = read_file(uploaded_file)

    if file_text and len(file_text.strip()) > 0:
        # Check if this is a new document
        if st.session_state.current_document != uploaded_file.name:
            st.session_state.current_document = uploaded_file.name
            st.session_state.logic_questions = None
            st.session_state.questions_loaded = False
            st.session_state.question_generation_attempts = 0
        
        st.success("✅ Document successfully loaded and processed.")
        
        # Display document stats
        word_count = len(file_text.split())
        char_count = len(file_text)
        st.info(f"📊 Document Stats: {word_count} words, {char_count} characters")
        
        # Show document preview
        with st.expander("📄 Document Preview"):
            st.text(file_text[:500] + "..." if len(file_text) > 500 else file_text)

        # Auto Summary
        with st.spinner("Generating document summary..."):
            try:
                summary = summarize_document(file_text)
                st.subheader("📄 Document Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")
                summary = "Summary generation failed."

        # Prepare vector store once
        with st.spinner("Preparing document for search..."):
            try:
                vector_store = prepare_vector_store(file_text)
                st.success("✅ Document indexed successfully")
            except Exception as e:
                st.error(f"Error preparing vector store: {str(e)}")
                st.stop()

        # Interaction modes
        mode = st.radio("Choose Interaction Mode:", ["Ask Anything", "Challenge Me"])

        if mode == "Ask Anything":
            st.subheader("🔎 Ask Anything About the Document")
            user_question = st.text_input("Enter your question:")

            if user_question:
                with st.spinner("Finding answer..."):
                    try:
                        answer, references = qa_chain(vector_store, user_question)

                        st.markdown("**Answer:**")
                        st.write(answer)

                        if references:
                            st.markdown("**Supporting Snippets:**")
                            for i, ref in enumerate(references, 1):
                                st.markdown(f"**Reference {i}:**")
                                st.code(ref, language="markdown")
                        else:
                            st.info("No supporting references found.")
                            
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")

        elif mode == "Challenge Me":
            st.subheader("🧠 Challenge Me")
            
            # Show attempt counter
            if st.session_state.question_generation_attempts > 0:
                st.info(f"Generation attempts: {st.session_state.question_generation_attempts}")

            # Generate questions button
            if not st.session_state.questions_loaded:
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Generate Questions", type="primary"):
                        st.session_state.question_generation_attempts += 1
                        with st.spinner("Generating logic-based questions..."):
                            try:
                                # Show more detailed progress
                                progress_placeholder = st.empty()
                                progress_placeholder.info("🔄 Analyzing document content...")
                                
                                questions = generate_logic_questions(file_text)
                                
                                if questions:
                                    st.session_state.logic_questions = questions
                                    st.session_state.questions_loaded = True
                                    progress_placeholder.success("✅ Questions generated successfully!")
                                else:
                                    progress_placeholder.error("❌ Failed to generate questions")
                                    
                            except Exception as e:
                                st.error(f"Error generating questions: {str(e)}")
                                st.session_state.logic_questions = []
                
                with col2:
                    if st.button("🔍 Debug Mode", help="Show detailed generation process"):
                        st.session_state.question_generation_attempts += 1
                        with st.spinner("Generating questions with debug info..."):
                            try:
                                # Capture debug output
                                import sys
                                from io import StringIO
                                
                                # Redirect stdout to capture print statements
                                old_stdout = sys.stdout
                                sys.stdout = captured_output = StringIO()
                                
                                questions = generate_logic_questions(file_text)
                                
                                # Restore stdout
                                sys.stdout = old_stdout
                                debug_output = captured_output.getvalue()
                                
                                # Show debug info
                                with st.expander("🔧 Debug Information"):
                                    st.code(debug_output, language="text")
                                
                                if questions:
                                    st.session_state.logic_questions = questions
                                    st.session_state.questions_loaded = True
                                    st.success("✅ Questions generated successfully!")
                                else:
                                    st.error("❌ Failed to generate questions")
                                    
                            except Exception as e:
                                sys.stdout = old_stdout
                                st.error(f"Error in debug mode: {str(e)}")

            # Display questions if loaded
            if st.session_state.questions_loaded and st.session_state.logic_questions:
                logic_questions = st.session_state.logic_questions

                st.markdown("### 💡 Answer These Questions")
                
                # Show question quality indicator
                if len(logic_questions) >= 3:
                    st.success(f"✅ {len(logic_questions)} questions generated successfully!")
                else:
                    st.warning(f"⚠️ Only {len(logic_questions)} questions generated (fallback mode)")

                # Optional: Raw question debug
                with st.expander("🧪 Raw Questions JSON"):
                    st.json(logic_questions)

                # Display questions
                for idx, q in enumerate(logic_questions):
                    with st.container():
                        st.markdown(f"**Question {idx + 1}:** {q['question']}")

                        # Create unique key for each question
                        selected = st.radio(
                            f"Choose your answer for Question {idx + 1}:",
                            q["options"],
                            key=f"q_{idx}_option"
                        )

                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if st.button(f"Check Answer {idx + 1}", key=f"check_{idx}"):
                                try:
                                    # Extract just the letter from the selected option
                                    selected_letter = selected[0]  # Get A, B, C, or D
                                    correct_answer = q["answer"].strip()
                                    
                                    if selected_letter == correct_answer:
                                        st.success("✅ Correct!")
                                    else:
                                        st.error(f"❌ Incorrect. The correct answer is: **{correct_answer}**")

                                    st.markdown(f"**Explanation:** {q['explanation']}")
                                    
                                except Exception as e:
                                    st.error(f"Error checking answer: {str(e)}")
                        
                        st.markdown("---")

            # Reset questions button
            if st.session_state.questions_loaded:
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🔄 Generate New Questions"):
                        st.session_state.logic_questions = None
                        st.session_state.questions_loaded = False
                        st.rerun()
                with col2:
                    if st.button("🗑️ Clear All"):
                        st.session_state.logic_questions = None
                        st.session_state.questions_loaded = False
                        st.session_state.question_generation_attempts = 0
                        st.rerun()

    else:
        st.error("❌ Could not extract text from the uploaded file. Please check the file format and try again.")
else:
    st.info("👆 Please upload a document to get started.")
    # Reset session state when no file is uploaded
    if st.session_state.current_document:
        st.session_state.current_document = None
        st.session_state.logic_questions = None
        st.session_state.questions_loaded = False
        st.session_state.question_generation_attempts = 0

# Footer
st.markdown("---")
st.markdown("🔧 **Built with Streamlit and LangChain**")