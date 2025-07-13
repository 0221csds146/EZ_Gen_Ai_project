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

# Page Configuration
st.set_page_config(
    page_title="EZ GenAI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
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

# Enhanced Custom CSS for better UI
def load_custom_css():
    st.markdown("""
    <style>
    /* Main Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.8rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-weight: 700;
    }
    
    .main-header p {
        color: #f0f0f0;
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: none;
    }
    
    .feature-card h3 {
        margin-top: 0;
        color: white;
    }
    
    .feature-card ul {
        list-style: none;
        padding: 0;
    }
    
    .feature-card li {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
    }
    
    .feature-card li:before {
        content: "✨";
        position: absolute;
        left: 0;
    }
    
    /* Stats Cards */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        flex: 1;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Enhanced Highlighting */
    .highlight {
        background: linear-gradient(90deg, #fff176 0%, #ffeb3b 100%);
        color: #000;
        padding: 3px 6px;
        border-radius: 4px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Enhanced Source Snippets */
    .source-snippet {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #212529;
        border-left: 4px solid #007bff;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 0.95em;
        line-height: 1.6;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    /* Enhanced Conversation Bubbles */
    .conversation-bubble {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        color: #0d47a1;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1976d2;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .conversation-bubble strong {
        color: #0d47a1;
    }

    /* Memory Indicator */
    .memory-indicator {
        background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%);
        color: #1b5e20;
        border: 2px solid #4caf50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.95em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Relevance Badge */
    .relevance-badge {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Mode Selection Enhancement */
    .mode-selector {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #ff9800;
    }
    
    /* Success/Error Messages */
    .success-message {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .error-message {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    /* Question Cards */
    .question-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border: 2px solid #ff9800;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .question-number {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    .document-preview {
    background: linear-gradient(135deg, #f5f5f5 0%, #eeeeee 100%);
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Courier New', monospace;
    font-size: 0.9em;
    max-height: 300px;
    overflow-y: auto;
    color: black; /* 👈 Add this line */
}
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #37474f 0%, #263238 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 3rem;
        text-align: center;
    }
    
    /* Animated Elements */
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .stats-container {
            flex-direction: column;
        }
        
        .stat-card {
            margin-bottom: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Enhanced answer display with highlighting
def display_enhanced_answer(result):
    """Display answer with highlighting and source snippets"""
    
    # Main answer with enhanced styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); 
                padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                border-left: 4px solid #4caf50;">
        <h3 style="color: #2e7d32; margin-top: 0;">📝 Answer</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.write(result["answer"])
    
    # Supporting quotes with highlighting
    if result.get("supporting_quotes"):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                    border-left: 4px solid #ff9800;">
            <h3 style="color: #ef6c00; margin-top: 0;">💡 Supporting Quotes</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for i, quote in enumerate(result["supporting_quotes"], 1):
            st.markdown(f"""
            <div class="source-snippet">
                <strong>Quote {i}:</strong> <em>"{quote}"</em>
            </div>
            """, unsafe_allow_html=True)
    
    # Highlighted source snippets
    if result.get("highlighted_sources"):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                    padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                    border-left: 4px solid #1976d2;">
            <h3 style="color: #0d47a1; margin-top: 0;">📚 Source Snippets with Highlighting</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for i, source in enumerate(result["highlighted_sources"], 1):
            relevance_score = source.get("relevance_score", 0)
            
            # Create relevance badge
            relevance_badge = f'<span class="relevance-badge">🎯 {relevance_score} matches</span>' if relevance_score > 0 else ""
            
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
                    st.caption(f"📊 Chunk ID: {metadata.get('chunk_id', 'N/A')} | Length: {metadata.get('chunk_length', 'N/A')} chars")

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
            st.caption(f"💭 Remembering {num_qa_pairs} previous Q&A pairs")

# Display conversation history
def display_conversation_history(conversation_history, show_all=False, max_recent=3):
    """Display formatted conversation history with configurable options"""
    if conversation_history:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); 
                    padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                    border-left: 4px solid #9c27b0;">
            <h3 style="color: #6a1b9a; margin-top: 0;">📜 Conversation History</h3>
        </div>
        """, unsafe_allow_html=True)
        
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
        st.caption(f"📊 Total conversations: {len(conversation_history)} | Showing: {len(conversations_to_show)}")
        
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

# Load custom CSS
load_custom_css()

# Main Header
st.markdown("""
<div class="main-header">
    <h1>🤖 EZ GenAI Research Assistant</h1>
    <p>Your AI-powered document analysis and Q&A companion</p>
</div>
""", unsafe_allow_html=True)

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

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""
    <div class="feature-card">
        <h3>🚀 Features</h3>
        <ul>
            <li>📄 Document Upload & Analysis</li>
            <li>🤖 AI-Powered Q&A</li>
            <li>💬 Conversational Interface</li>
            <li>🧠 Logic-Based Challenges</li>
            <li>📊 Smart Highlighting</li>
            <li>🔄 Memory-Aware Chat</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Memory & Settings
    st.markdown("### 🧠 Memory & Settings")
    
    # Memory status
    if st.session_state.conversation_chain:
        memory_length = len(st.session_state.conversation_chain.memory.chat_memory.messages)
        
        # Stats display
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-number">{memory_length // 2}</div>
                <div class="stat-label">Q&A Pairs</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Memory", use_container_width=True):
            st.session_state.conversation_chain.clear_memory()
            st.session_state.conversation_history = []
            if 'current_memory_result' in st.session_state:
                del st.session_state.current_memory_result
            if 'current_memory_question' in st.session_state:
                del st.session_state.current_memory_question
            st.success("Memory cleared!")
            st.rerun()
    
    # Display settings
    st.markdown("### ⚙️ Display Settings")
    max_sources = st.slider("Max Sources to Show", 1, 3, 3)
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        1. **Upload** your document (PDF, TXT)
        2. **Read** the auto-generated summary
        3. **Ask** questions about the content
        4. **Challenge** yourself with logic questions
        5. **Explore** highlighted source passages
        """)
    
    # Links
    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("[📚  Documentation](https://docs.streamlit.io)")
    st.markdown("[🐙  GitHub](https://github.com/0221csds146/EZ_Gen_Ai_project)")
    st.markdown("[💬  Support](mailto:kumarr22470@gmail.com)")

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    # File upload section with enhanced styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); 
                padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                border-left: 4px solid #4caf50;">
        <h2 style="color: #2e7d32; margin-top: 0;">📄 Document Upload</h2>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a document to analyze",
        type=["pdf", "txt"],
        help="Upload PDF or TXT files for analysis"
    )

with col2:
    # Quick stats or tips
    if st.session_state.vector_store is not None:
        st.markdown("""
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-number">✅</div>
                <div class="stat-label">Document Ready</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-number">{len(st.session_state.conversation_history)}</div>
                <div class="stat-label">Questions Asked</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if uploaded_file:
    # Read file content with loading indicator
    with st.spinner("📖 Reading document..."):
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
        
        # Success message
        st.markdown("""
        <div class="success-message">
            <strong>✅ Document successfully loaded and processed!</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Display document stats
        word_count = len(file_text.split())
        char_count = len(file_text)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Words", f"{word_count:,}")
        with col2:
            st.metric("🔤 Characters", f"{char_count:,}")
        with col3:
            st.metric("📄 Pages", f"~{word_count//250}")
        
        # Show document preview
        with st.expander("📄 Document Preview", expanded=False):
            st.markdown(f"""
            <div class="document-preview">
                {file_text[:500]}{'...' if len(file_text) > 500 else ''}
            </div>
            """, unsafe_allow_html=True)

        # Auto Summary
        with st.spinner("🤖 Generating intelligent summary..."):
            try:
                summary = summarize_document(file_text)
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); 
                            padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                            border-left: 4px solid #ff9800;">
                    <h2 style="color: #ef6c00; margin-top: 0;">📄 Document Summary</h2>
                </div>
                """, unsafe_allow_html=True)
                st.write(summary)
            except Exception as e:
                st.markdown("""
                <div class="error-message">
                    <strong>❌ Error generating summary:</strong> {str(e)}
                </div>
                """, unsafe_allow_html=True)
                summary = "Summary generation failed."

        # Prepare vector store once
        if st.session_state.vector_store is None:
            with st.spinner("🔍 Preparing document for intelligent search..."):
                try:
                    vector_store = prepare_vector_store(file_text)
                    st.session_state.vector_store = vector_store
                    st.markdown("""
                    <div class="success-message">
                        <strong>✅ Document indexed successfully!</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Initialize enhanced conversation chain
                    if st.session_state.conversation_chain is None:
                        st.session_state.conversation_chain = EnhancedConversationalChain(vector_store)
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="error-message">
                        <strong>❌ Error preparing vector store:</strong> {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
                    st.stop()

        # Interaction modes with enhanced styling
        st.markdown("---")
        st.markdown("""
        <div class="mode-selector">
            <h3 style="color: #e65100; margin-top: 0;">🎯 Choose Your Interaction Mode</h3>
        </div>
        """, unsafe_allow_html=True)
        
        mode = st.radio(
            "Select mode:",
            ["Ask Anything", "Memory Chat", "Challenge Me"],
            horizontal=True
        )

        if mode == "Ask Anything":
            st.markdown("""
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                        padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                        border-left: 4px solid #1976d2;">
                <h2 style="color: #0d47a1; margin-top: 0;">🔎 Ask Anything About the Document</h2>
                <p style="color: #1565c0; margin-bottom: 0;"><em>✨ Enhanced with answer highlighting and source snippets</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            user_question = st.text_input("Enter your question:", placeholder="What is the main topic of this document?")

            if user_question:
                with st.spinner("🤖 Finding answer with intelligent highlighting..."):
                    try:
                        # Use enhanced QA with highlighting
                        result = qa_chain_with_highlighting(st.session_state.vector_store, user_question)
                        
                        # Display enhanced answer
                        display_enhanced_answer(result)
                            
                    except Exception as e:
                        st.markdown(f"""
                        <div class="error-message">
                            <strong>❌ Error processing question:</strong> {str(e)}
                        </div>
                        """, unsafe_allow_html=True)

        elif mode == "Memory Chat":
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); 
                        padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                        border-left: 4px solid #9c27b0;">
                <h2 style="color: #6a1b9a; margin-top: 0;">🧠 Memory-Aware Conversation</h2>
                <p style="color: #7b1fa2; margin-bottom: 0;"><em>✨ Ask follow-up questions that refer to previous interactions</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display memory context
            display_memory_context(st.session_state.conversation_chain)
            
            # Chat interface
            user_question = st.text_input("Enter your question (can refer to previous answers):", 
                                        placeholder="Can you expand on that previous point?")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("💬 Ask Question", type="primary", use_container_width=True):
                    if user_question:
                        with st.spinner("🤖 Processing with memory context..."):
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
