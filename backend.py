from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import json
import re
from groq_llm import get_groq_llm
from prompts import SUMMARY_PROMPT, LOGIC_QUESTION_GEN_PROMPT, EVALUATE_RESPONSE_PROMPT,ENHANCED_QA_PROMPT

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Enhanced prompt for answer highlighting


# 1. Create Vector Store with metadata
def prepare_vector_store(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    chunks = text_splitter.split_text(raw_text)
    docs = []
    
    for i, chunk in enumerate(chunks):
        # Add metadata to each chunk for better tracking
        doc = Document(
            page_content=chunk,
            metadata={
                "chunk_id": i,
                "chunk_length": len(chunk),
                "start_char": raw_text.find(chunk),
                "end_char": raw_text.find(chunk) + len(chunk)
            }
        )
        docs.append(doc)
    
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# 2. Generate Auto Summary
def summarize_document(content):
    content = content[:5000]
    llm = get_groq_llm()
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(SUMMARY_PROMPT))
    return chain.run(content=content)

# 3. Enhanced QA Chain with Answer Highlighting
def qa_chain_with_highlighting(vector_store, query, conversation_memory=None):
    """Enhanced QA with answer highlighting and optional memory"""
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Get more context
    llm = get_groq_llm()
    
    # Get relevant documents
    relevant_docs = retriever.get_relevant_documents(query)
    
    # Combine context from all relevant documents
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Create enhanced prompt
    prompt = PromptTemplate.from_template(ENHANCED_QA_PROMPT)
    
    # Add conversation memory if provided
    if conversation_memory:
        # Get conversation history
        history = conversation_memory.chat_memory.messages
        if history:
            conversation_context = "\n".join([
                f"Previous Q: {msg.content}" if msg.type == "human" else f"Previous A: {msg.content}"
                for msg in history[-4:]  # Last 2 Q&A pairs
            ])
            context = f"Previous conversation:\n{conversation_context}\n\nCurrent context:\n{context}"
    
    # Generate answer
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(context=context, question=query)
    
    # Parse response to separate answer and quotes
    answer_parts = response.split("SUPPORTING_QUOTES:")
    main_answer = answer_parts[0].replace("ANSWER:", "").strip()
    
    supporting_quotes = []
    if len(answer_parts) > 1:
        quotes_text = answer_parts[1].strip()
        # Extract quotes (lines that start with quotes or contain quoted text)
        for line in quotes_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('"') or '"' in line):
                supporting_quotes.append(line.strip('"'))
    
    # Find and highlight source snippets
    highlighted_sources = []
    for doc in relevant_docs:
        snippet = doc.page_content
        metadata = doc.metadata
        
        # Try to find overlapping content with supporting quotes
        relevance_score = 0
        for quote in supporting_quotes:
            if quote.lower() in snippet.lower():
                relevance_score += 1
        
        highlighted_sources.append({
            "content": snippet,
            "metadata": metadata,
            "relevance_score": relevance_score,
            "highlighted_parts": supporting_quotes
        })
    
    # Sort by relevance
    highlighted_sources.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return {
        "answer": main_answer,
        "supporting_quotes": supporting_quotes,
        "highlighted_sources": highlighted_sources[:3],  # Top 3 most relevant
        "all_sources": [doc.page_content for doc in relevant_docs]
    }

# 4. Memory-aware Conversational Chain
class EnhancedConversationalChain:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
        )
        self.llm = get_groq_llm()
        
    def ask_question(self, question):
        """Ask a question with memory context"""
        result = qa_chain_with_highlighting(
            self.vector_store, 
            question, 
            self.memory
        )
        
        # Save to memory
        self.memory.save_context(
            {"input": question},
            {"answer": result["answer"]}
        )
        
        return result
    
    def get_conversation_history(self):
        """Get formatted conversation history"""
        history = self.memory.chat_memory.messages
        formatted_history = []
        
        for i in range(0, len(history), 2):
            if i + 1 < len(history):
                q = history[i].content
                a = history[i + 1].content
                formatted_history.append({"question": q, "answer": a})
        
        return formatted_history
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()

# 5. Text highlighting utility
def highlight_text(text, phrases_to_highlight):
    """Highlight specific phrases in text"""
    if not phrases_to_highlight:
        return text
    
    highlighted_text = text
    for phrase in phrases_to_highlight:
        if phrase and phrase.strip():
            # Use case-insensitive matching
            pattern = re.escape(phrase.strip())
            highlighted_text = re.sub(
                pattern, 
                f"**{phrase.strip()}**", 
                highlighted_text, 
                flags=re.IGNORECASE
            )
    
    return highlighted_text

# 6. Legacy QA function for backward compatibility
def qa_chain(vector_store, query):
    """Legacy QA function - maintained for backward compatibility"""
    result = qa_chain_with_highlighting(vector_store, query)
    return result["answer"], result["all_sources"][:3]

# Helper function to clean JSON response
def clean_json_response(response):
    """Clean and extract JSON from LLM response"""
    print(f"🔍 Original response: {response}")
    
    # Try to find JSON array boundaries
    start_idx = response.find('[')
    end_idx = response.rfind(']')
    
    if start_idx == -1 or end_idx == -1:
        print("❌ No JSON array found in response")
        return None
    
    # Extract JSON portion
    json_str = response[start_idx:end_idx + 1]
    
    # Clean up common issues
    json_str = json_str.replace("'", '"')  # Replace single quotes
    json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas in objects
    json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # Remove control characters
    
    print(f"🧹 Cleaned JSON: {json_str}")
    return json_str

# Helper function to validate question format
def validate_question_format(question):
    """Validate if a question has the correct format"""
    required_keys = ["question", "options", "answer", "explanation"]
    
    if not isinstance(question, dict):
        return False
    
    if not all(key in question for key in required_keys):
        return False
    
    if not isinstance(question["options"], list) or len(question["options"]) != 4:
        return False
    
    if not isinstance(question["answer"], str):
        return False
    
    return True

# Helper function to generate fallback questions with document context
def generate_fallback_questions(content):
    """Generate fallback questions when LLM fails"""
    # Try to extract key information from content
    words = content.split()
    first_sentence = ' '.join(words[:20]) if len(words) > 20 else content
    
    # Try to identify document type
    doc_type = "technical document"
    if any(word in content.lower() for word in ['history', 'historical', 'century', 'year']):
        doc_type = "historical document"
    elif any(word in content.lower() for word in ['science', 'research', 'study', 'experiment']):
        doc_type = "scientific document"
    elif any(word in content.lower() for word in ['code', 'function', 'algorithm', 'programming']):
        doc_type = "technical document"
    
    return [
        {
            "question": f"What is the primary focus of this {doc_type}?",
            "options": [
                "A) Providing step-by-step instructions",
                "B) Explaining concepts and information", 
                "C) Telling a story",
                "D) Listing facts without context"
            ],
            "answer": "B",
            "explanation": "Documents typically aim to explain concepts and provide information to help readers understand the subject matter."
        },
        {
            "question": "What reading strategy would be most effective for this content?",
            "options": [
                "A) Skimming quickly for keywords",
                "B) Reading only the conclusion",
                "C) Careful analysis and comprehension",
                "D) Memorizing without understanding"
            ],
            "answer": "C",
            "explanation": "Effective reading requires careful analysis and comprehension to fully understand the material."
        },
        {
            "question": "How should complex information in documents be approached?",
            "options": [
                "A) Ignore difficult sections",
                "B) Break down into smaller parts for analysis",
                "C) Accept without questioning",
                "D) Focus only on familiar terms"
            ],
            "answer": "B",
            "explanation": "Breaking complex information into smaller, manageable parts allows for better understanding and analysis."
        }
    ]

# 7. Improved Challenge Me: Logic-Based Questions
def generate_logic_questions(content):
    """Generate logic-based questions with improved error handling"""
    
    # Limit content to avoid token limits
    content = content[:3000]
    
    # Use lower temperature for more consistent output
    llm = get_groq_llm(temperature=0.1)
    
    # Use improved prompt
    prompt = PromptTemplate.from_template(LOGIC_QUESTION_GEN_PROMPT)
    chain = LLMChain(llm=llm, prompt=prompt)

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            print(f"🔄 Attempt {attempt + 1} to generate questions...")
            
            # Generate response
            response = chain.run(context=content)
            print(f"📝 Raw LLM Response:\n{response}")

            # Clean the response
            cleaned_json = clean_json_response(response)
            if not cleaned_json:
                print("❌ Failed to extract JSON from response")
                continue

            # Parse JSON
            questions = json.loads(cleaned_json)
            print(f"✅ Parsed {len(questions)} questions")

            # Validate questions
            if isinstance(questions, list) and len(questions) > 0:
                valid_questions = []
                for i, q in enumerate(questions):
                    if validate_question_format(q):
                        # Clean up the answer format
                        answer = q["answer"].strip().upper()
                        # Extract just the letter
                        if answer in ['A', 'B', 'C', 'D']:
                            q["answer"] = answer
                            valid_questions.append(q)
                            print(f"✅ Question {i+1} validated")
                        else:
                            print(f"❌ Question {i+1} has invalid answer format: {answer}")
                    else:
                        print(f"❌ Question {i+1} failed validation")
                
                if len(valid_questions) >= 2:  # Accept if we have at least 2 good questions
                    print(f"🎉 Successfully generated {len(valid_questions)} valid questions!")
                    return valid_questions
                else:
                    print(f"⚠️ Only {len(valid_questions)} valid questions generated, need at least 2")

        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error on attempt {attempt + 1}: {e}")
            if cleaned_json:
                print(f"Problematic JSON: {cleaned_json[:200]}...")
        except Exception as e:
            print(f"❌ Unexpected error on attempt {attempt + 1}: {e}")
    
    # If all attempts fail, use fallback
    print("🔄 All attempts failed, using fallback questions based on document content")
    return generate_fallback_questions(content)

# 8. Evaluate user's freeform answer to challenge question
def evaluate_user_response(document, question, response):
    llm = get_groq_llm()
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(EVALUATE_RESPONSE_PROMPT))
    return chain.run(context=document, question=question, response=response)

# 9. Legacy function for backward compatibility
def get_conversational_chain(vector_store: FAISS):
    """Legacy function - use EnhancedConversationalChain instead"""
    return EnhancedConversationalChain(vector_store)