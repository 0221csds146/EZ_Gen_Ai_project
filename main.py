import streamlit as st
import os
from backend import (
    prepare_vector_store,
    summarize_document,
    qa_chain,
    qa_chain_with_highlighting,  # New enhanced function
    generate_logic_questions,
    evaluate_user_response,
    get_conversational_chain,
    EnhancedConversationalChain,  # New enhanced class
    highlight_text  # New highlighting utility
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

# Custom CSS for better highlighting and UI
def load_custom_css():
    st.markdown("""
    <style>
    .highlight {
        background-color: #fff176; /* Brighter yellow for better contrast */
        color: #000;               /* Black text for visibility */
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }

    .source-snippet {
        background-color: #f1f3f4; /* Slightly darker gray */
        color: #000;               /* Black text */
        border-left: 4px solid #007bff;
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
    }

    .conversation-bubble {
        background-color: #e3f2fd;
        color: #000;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #1976d2;
    }

    .memory-indicator {
        background-color: #d0f0d0; /* Higher contrast light green */
        color: #000;               /* Black text */
        border: 1px solid #4caf50;
        border-radius: 5px;
        padding: 8px;
        margin: 5px 0;
        font-size: 0.9em;
    }

    .relevance-badge {
        background-color: #28a745;
        color: white;
        padding: 2px 6px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Enhanced answer display with highlighting
def display_enhanced_answer(result):
    """Display answer with highlighting and source snippets"""
    
    # Main answer
    st.markdown("### 📝 Answer:")
    st.write(result["answer"])
    
    # Supporting quotes with highlighting
    if result.get("supporting_quotes"):
        st.markdown("### 💡 Supporting Quotes:")
        for i, quote in enumerate(result["supporting_quotes"], 1):
            st.markdown(f"""
            <div class="source-snippet">
                <strong>Quote {i}:</strong> <em>"{quote}"</em>
            </div>
            """, unsafe_allow_html=True)
    
    # Highlighted source snippets
    if result.get("highlighted_sources"):
        st.markdown("### 📚 Source Snippets with Highlighting:")
        for i, source in enumerate(result["highlighted_sources"], 1):
            relevance_score = source.get("relevance_score", 0)
            
            # Create relevance badge
            relevance_badge = f'<span class="relevance-badge">{relevance_score} matches</span>' if relevance_score > 0 else ""
            
            with st.expander(
                f"📄 Source {i} " + (f"({relevance_score} matches)" if relevance_score > 0 else ""), 
                expanded=(i == 1)  # Expand first source by default
            ):
                # Highlight the relevant parts
                highlighted_content = highlight_text(
                    source["content"], 
                    source.get("highlighted_parts", [])
                )
                
                # Convert markdown bold to HTML highlighting
                highlighted_content = highlighted_content.replace('**', '<span class="highlight">').replace('**', '</span>')
                
                # Display with custom styling
                st.markdown(f"""
                <div class="source-snippet">
                    {relevance_badge}
                    <br><br>
                    {highlighted_content}
                </div>
                """, unsafe_allow_html=True)
                
                # Show metadata
                if source.get("metadata"):
                    metadata = source["metadata"]
                    st.caption(f"Chunk ID: {metadata.get('chunk_id', 'N/A')} | Length: {metadata.get('chunk_length', 'N/A')} chars")

# Memory context display
def display_memory_context(conversation_chain):
    """Display conversation memory context"""
    if hasattr(conversation_chain, 'memory') and conversation_chain.memory.chat_memory.messages:
        messages = conversation_chain.memory.chat_memory.messages
        if messages:
            st.markdown("""
            <div class="memory-indicator">
                🧠 <strong>Memory Active:</strong> I can reference our previous conversation.
            </div>
            """, unsafe_allow_html=True)
            
            # Show memory stats
            num_qa_pairs = len(messages) // 2
            st.caption(f"Remembering {num_qa_pairs} previous Q&A pairs")

# Display conversation history
def display_conversation_history(conversation_history, show_all=False, max_recent=3):
    """Display formatted conversation history with configurable options"""
    if conversation_history:
        st.markdown("### 📜 Conversation History:")
        
        # User control for display options
        col1, col2 = st.columns(2)
        with col1:
            show_all_toggle = st.checkbox("Show all conversations", value=show_all)
        with col2:
            if not show_all_toggle:
                max_display = st.slider("Recent conversations to show", 1, 10, max_recent)
        
        # Determine which conversations to show
        if show_all_toggle:
            conversations_to_show = conversation_history
            start_index = 1
        else:
            conversations_to_show = conversation_history[-max_display:]
            start_index = max(1, len(conversation_history) - max_display + 1)
        
        # Show total count
        st.caption(f"Total conversations: {len(conversation_history)} | Showing: {len(conversations_to_show)}")
        
        # Display conversations
        for i, chat in enumerate(conversations_to_show):
            actual_index = start_index + i
            
            with st.expander(f"💬 Exchange {actual_index}", expanded=False):
                st.markdown(f"""
                <div class="conversation-bubble">
                    <strong>Q:</strong> {chat['question']}<br><br>
                    <strong>A:</strong> {chat['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Show supporting quotes if available
                if chat.get('supporting_quotes'):
                    st.markdown("**Supporting Quotes:**")
                    for quote in chat['supporting_quotes']:
                        st.caption(f"• \"{quote}\"")

# Streamlit UI
st.set_page_config(page_title="Smart Research Assistant", layout="wide")

# Load custom CSS
load_custom_css()

st.title("📚 Smart Assistant for Research Summarization")
st.markdown("### 🎯 New Features: Memory-Aware Conversations & Answer Highlighting")

# Initialize session state variables
if "logic_questions" not in st.session_state:
    st.session_state.logic_questions = None
if "questions_loaded" not in st.session_state:
    st.session_state.questions_loaded = False
if "current_document" not in st.session_state:
    st.session_state.current_document = None
if "question_generation_attempts" not in st.session_state:
    st.session_state.question_generation_attempts = 0
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_memory_result" not in st.session_state:
    st.session_state.current_memory_result = None
if "current_memory_question" not in st.session_state:
    st.session_state.current_memory_question = None

# Sidebar for memory management
with st.sidebar:
    st.header("🧠 Memory & Settings")
    
    # Memory status
    if st.session_state.conversation_chain:
        memory_length = len(st.session_state.conversation_chain.memory.chat_memory.messages)
        st.info(f"💭 Memory: {memory_length // 2} Q&A pairs")
        
        if st.button("🗑️ Clear Memory"):
            st.session_state.conversation_chain.clear_memory()
            st.session_state.conversation_history = []
            if 'current_memory_result' in st.session_state:
                del st.session_state.current_memory_result
            if 'current_memory_question' in st.session_state:
                del st.session_state.current_memory_question
            st.success("Memory cleared!")
            st.rerun()
    
    # Display settings
    st.subheader("Display Settings")
    max_sources = st.slider("Max Sources to Show", 1,3)

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
            st.session_state.conversation_chain = None
            st.session_state.conversation_history = []
            st.session_state.vector_store = None
            if 'current_memory_result' in st.session_state:
                del st.session_state.current_memory_result
            if 'current_memory_question' in st.session_state:
                del st.session_state.current_memory_question
        
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
        if st.session_state.vector_store is None:
            with st.spinner("Preparing document for search..."):
                try:
                    vector_store = prepare_vector_store(file_text)
                    st.session_state.vector_store = vector_store
                    st.success("✅ Document indexed successfully")
                    
                    # Initialize enhanced conversation chain
                    if st.session_state.conversation_chain is None:
                        st.session_state.conversation_chain = EnhancedConversationalChain(vector_store)
                    
                except Exception as e:
                    st.error(f"Error preparing vector store: {str(e)}")
                    st.stop()

        # Interaction modes
        st.markdown("---")
        mode = st.radio("Choose Interaction Mode:", ["Ask Anything", "Memory Chat", "Challenge Me"])

        if mode == "Ask Anything":
            st.subheader("🔎 Ask Anything About the Document")
            st.markdown("*✨ Now with answer highlighting and source snippets*")
            
            user_question = st.text_input("Enter your question:")

            if user_question:
                with st.spinner("Finding answer with highlighting..."):
                    try:
                        # Use enhanced QA with highlighting
                        result = qa_chain_with_highlighting(st.session_state.vector_store, user_question)
                        
                        # Display enhanced answer
                        display_enhanced_answer(result)
                            
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")

        elif mode == "Memory Chat":
            st.subheader("🧠 Memory-Aware Conversation")
            st.markdown("*✨ Ask follow-up questions that refer to previous interactions*")
            
            # Display memory context
            display_memory_context(st.session_state.conversation_chain)
            
            # Chat interface
            user_question = st.text_input("Enter your question (can refer to previous answers):")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("💬 Ask Question", type="primary"):
                    if user_question:
                        with st.spinner("Processing with memory context..."):
                            try:
                                # Use enhanced conversational chain
                                result = st.session_state.conversation_chain.ask_question(user_question)
                                
                                # Store the current result to display
                                st.session_state.current_memory_result = result
                                st.session_state.current_memory_question = user_question
                                
                                # Add to display history
                                st.session_state.conversation_history.append({
                                    "question": user_question,
                                    "answer": result["answer"],
                                    "supporting_quotes": result.get("supporting_quotes", []),
                                    "highlighted_sources": result.get("highlighted_sources", [])
                                })
                                
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error in memory chat: {str(e)}")
            
            with col2:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.conversation_chain.clear_memory()
                    st.session_state.conversation_history = []
                    if 'current_memory_result' in st.session_state:
                        del st.session_state.current_memory_result
                    if 'current_memory_question' in st.session_state:
                        del st.session_state.current_memory_question
                    st.success("Chat history cleared!")
                    st.rerun()
            
            # Display current answer if available
            if 'current_memory_result' in st.session_state:
                st.markdown("### 📝 Current Answer:")
                if st.session_state.get("current_memory_result") is not None:
                    display_enhanced_answer(st.session_state.current_memory_result)
                st.markdown("---")
            
            # Display conversation history
            display_conversation_history(st.session_state.conversation_history)

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
                

            # Display questions if loaded
            if st.session_state.questions_loaded and st.session_state.logic_questions:
                logic_questions = st.session_state.logic_questions

                st.markdown("### 💡 Answer These Questions")
                
                # Show question quality indicator
                if len(logic_questions) >= 3:
                    st.success(f"✅ {len(logic_questions)} questions generated successfully!")
                else:
                    st.warning(f"⚠️ Only {len(logic_questions)} questions generated (fallback mode)")

                # Optional: Raw question info
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
        st.session_state.conversation_chain = None
        st.session_state.conversation_history = []
        st.session_state.vector_store = None
        if 'current_memory_result' in st.session_state:
            del st.session_state.current_memory_result
        if 'current_memory_question' in st.session_state:
            del st.session_state.current_memory_question

# Footer
st.markdown("---")
st.markdown("🔧 **Built with Streamlit and LangChain** | ✨ **Enhanced with Memory & Highlighting**")