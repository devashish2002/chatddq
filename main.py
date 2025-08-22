import streamlit as st
import os
import tempfile
import pandas as pd
import re
import unicodedata
import nltk

# Try different langchain imports based on version
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
except ImportError:
    # Fallback for older versions
    try:
        from langchain.chat_models import ChatOpenAI
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.vectorstores import FAISS
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
        from langchain.chains import RetrievalQA, ConversationalRetrievalChain
        from langchain.prompts import ChatPromptTemplate
        from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
        InMemoryVectorStore = FAISS
    except ImportError as e:
        st.error(f"LangChain import error: {e}")
        st.stop()

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    st.warning("PyMuPDF not available. Install with: pip install PyMuPDF")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    st.warning("pdfplumber not available. Install with: pip install pdfplumber")

nltk.download('punkt_tab')

# Simple memory management class
class SimpleMemory:
    def __init__(self, key: str = "memory_messages"):
        self.key = key
        if key not in st.session_state:
            st.session_state[key] = []

    def add_exchange(self, question: str, answer: str):
        """Add a question-answer exchange to memory"""
        st.session_state[self.key].append({
            "type": "human",
            "content": question
        })
        st.session_state[self.key].append({
            "type": "ai", 
            "content": answer
        })

    def get_messages(self):
        """Get all messages"""
        return st.session_state[self.key] if self.key in st.session_state else []

    def clear(self):
        """Clear all memory"""
        st.session_state[self.key] = []

    def format_for_prompt(self):
        """Format ALL memory for inclusion in prompt"""
        messages = self.get_messages()
        if not messages:
            return ""
        
        formatted_history = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i]
                ai_msg = messages[i + 1]
                formatted_history.append(f"Student: {human_msg['content']}")
                formatted_history.append(f"Tutor: {ai_msg['content']}")
        
        return "\n".join(formatted_history)

def clean_text(text):
    """Clean and normalize text extracted from PDFs"""
    if not text:
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters but keep newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Fix common LaTeX conversion issues
    text = text.replace('ﬁ', 'fi')  # Fix ligatures
    text = text.replace('ﬂ', 'fl')
    text = text.replace('ﬀ', 'ff')
    text = text.replace('ﬃ', 'ffi')
    text = text.replace('ﬄ', 'ffl')
    
    # Remove excessive line breaks but preserve paragraph structure
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text.strip()

def extract_text_with_pymupdf(file_path):
    """Extract text using PyMuPDF (fitz) - often better for LaTeX PDFs"""
    if not PYMUPDF_AVAILABLE:
        return None
    try:
        doc = fitz.open(file_path)
        texts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            cleaned_text = clean_text(text)
            if cleaned_text:  # Only add non-empty pages
                texts.append(Document(
                    page_content=cleaned_text,
                    metadata={"source": file_path, "page": page_num + 1}
                ))
        doc.close()
        return texts
    except Exception as e:
        st.error(f"PyMuPDF extraction failed: {str(e)}")
        return None

def extract_text_with_pdfplumber(file_path):
    """Extract text using pdfplumber - good for structured documents"""
    if not PDFPLUMBER_AVAILABLE:
        return None
    try:
        texts = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                cleaned_text = clean_text(text)
                if cleaned_text:  # Only add non-empty pages
                    texts.append(Document(
                        page_content=cleaned_text,
                        metadata={"source": file_path, "page": page_num + 1}
                    ))
        return texts
    except Exception as e:
        st.error(f"PDFplumber extraction failed: {str(e)}")
        return None

def extract_pdf_text(file_path, method="auto"):
    """Extract text from PDF using multiple methods as fallback"""
    
    if method == "auto":
        # Try PyMuPDF first (often best for LaTeX)
        if PYMUPDF_AVAILABLE:
            docs = extract_text_with_pymupdf(file_path)
            if docs and len(docs) > 0:
                # Check if extraction was successful by examining text quality
                sample_text = docs[0].page_content[:500] if docs else ""
                if len(sample_text) > 50 and not is_corrupted_text(sample_text):
                    st.success("✅ Successfully extracted text using PyMuPDF")
                    return docs
        
        # Try pdfplumber as second option
        if PDFPLUMBER_AVAILABLE:
            st.warning("PyMuPDF didn't work well, trying pdfplumber...")
            docs = extract_text_with_pdfplumber(file_path)
            if docs and len(docs) > 0:
                sample_text = docs[0].page_content[:500] if docs else ""
                if len(sample_text) > 50 and not is_corrupted_text(sample_text):
                    st.success("✅ Successfully extracted text using pdfplumber")
                    return docs
        
        # Fall back to PyPDFLoader
        st.warning("Trying PyPDFLoader as fallback...")
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            # Clean the text from PyPDFLoader too
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
            if docs and len(docs) > 0:
                sample_text = docs[0].page_content[:500] if docs else ""
                if not is_corrupted_text(sample_text):
                    st.success("✅ Successfully extracted text using PyPDFLoader")
                    return docs
        except Exception as e:
            st.error(f"PyPDFLoader failed: {str(e)}")
        
        # Try UnstructuredPDFLoader as last resort
        st.warning("Trying UnstructuredPDFLoader as last resort...")
        try:
            loader = UnstructuredPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
            if docs and len(docs) > 0:
                st.success("✅ Successfully extracted text using UnstructuredPDFLoader")
                return docs
        except Exception as e:
            st.error(f"UnstructuredPDFLoader failed: {str(e)}")
    
    return None

def is_corrupted_text(text):
    """Check if text appears to be corrupted or encoded improperly"""
    if not text:
        return True
    
    # Check for excessive special characters
    special_char_ratio = sum(1 for c in text if not c.isprintable() and c not in '\n\t ') / len(text)
    if special_char_ratio > 0.3:  # More than 30% special characters
        return True
    
    # Check for patterns that suggest encoding issues
    corrupted_patterns = [
        r'[^\x00-\x7F]{10,}',  # Long sequences of non-ASCII
        r'[\x00-\x08\x0E-\x1F]{5,}',  # Control characters
        r'\?{5,}',  # Multiple question marks (encoding failures)
    ]
    
    for pattern in corrupted_patterns:
        if re.search(pattern, text):
            return True
    
    return False

def format_chat_history_for_prompt(messages, max_exchanges=5):
    """Format simple message list for inclusion in prompt"""
    if not messages:
        return ""
    
    # Take only the last max_exchanges conversations
    recent_messages = messages[-max_exchanges*2:] if len(messages) > max_exchanges*2 else messages
    
    formatted_history = []
    for i in range(0, len(recent_messages), 2):
        if i + 1 < len(recent_messages):
            human_msg = recent_messages[i]
            ai_msg = recent_messages[i + 1]
            formatted_history.append(f"Student: {human_msg['content']}")
            formatted_history.append(f"Tutor: {ai_msg['content']}")
    
    return "\n".join(formatted_history)

def detect_problem_indicators(text):
    """Detect if text contains numerical problems or examples"""
    problem_indicators = [
        r'example\s+\d+',
        r'problem\s+\d+',
        r'exercise\s+\d+',
        r'question\s+\d+',
        r'solution:?',
        r'answer:?',
        r'calculate',
        r'find\s+the',
        r'solve\s+for',
        r'\$[0-9,]+',  # Dollar amounts
        r'\d+%',  # Percentages
        r'\d+\.\d+',  # Decimal numbers
        r'given:?',
        r'let\s+\w+\s*=',
        r'if\s+.*\s+then',
    ]
    
    score = 0
    for pattern in problem_indicators:
        matches = re.findall(pattern, text, re.IGNORECASE)
        score += len(matches)
    
    return score > 2  # Threshold for considering it a problem/example

def create_examples_retriever(docs, embeddings):
    """Create a specialized retriever for examples and problems"""
    # Filter documents that likely contain examples/problems
    example_docs = []
    for doc in docs:
        if detect_problem_indicators(doc.page_content):
            # Mark as example document
            doc.metadata['content_type'] = 'example'
            example_docs.append(doc)
    
    if example_docs:
        try:
            examples_vectorstore = InMemoryVectorStore.from_documents(
                documents=example_docs, 
                embedding=embeddings
            )
        except:
            from langchain.vectorstores import FAISS
            examples_vectorstore = FAISS.from_documents(example_docs, embeddings)
        
        return examples_vectorstore.as_retriever(search_kwargs={"k": 3})
    else:
        return None

def create_conversational_chain(retriever, llm, examples_retriever=None):
    """Create a conversational retrieval chain with simple memory"""
    
    # Initialize memory
    memory = SimpleMemory("conversation_memory")
    
    # Enhanced system prompt with memory awareness and examples focus
    system_prompt = (
        "You are an AI tutor specializing in academic documents. "
        "Use the retrieved context and conversation history to provide helpful, educational responses. "
        "Your role is to:\n"
        "1. Answer questions based on the document context\n"
        "2. Build upon previous conversations to provide continuity\n"
        "3. Clarify concepts when the student seems confused\n"
        "4. Suggest related topics or follow-up questions\n"
        "5. Reference previous topics when relevant\n\n"
        "If mathematical formulas or technical terms appear garbled, infer meaning from context. "
        "If you don't know something, say so clearly. "
        "Keep responses educational and engaging.\n\n"
        "Previous conversation:\n{chat_history}\n\n"
        "Retrieved context:\n{context}\n\n"
        "Current question: {question}\n\n"
        "Provide a comprehensive answer and suggest 2-3 relevant follow-up questions."
    )
    
    # Examples-focused system prompt
    examples_prompt = (
        "You are an AI tutor specialized in finding and explaining practical examples and numerical problems. "
        "Only provide practical examples and numerical problems when user asks questions on a specific mathematical topic or concept."
        "Don't output anything if the question is very general or just asks summary of the document.s"
        "Focus specifically on:\n"
        "1. Numerical problems with step-by-step solutions\n"
        "2. Practical examples that illustrate concepts\n"
        "3. Calculation methods and formulas\n"
        "4. Real-world applications\n"
        "5. Exercise problems and their solutions\n\n"
        "Based on the context below, provide concrete examples and numerical problems related to: {question}\n\n"
        "Context with examples:\n{examples_context}\n\n"
        "Provide specific numerical examples, step-by-step calculations, and practice problems if available."
    )
    
    def conversational_rag_chain(inputs):
        """Main conversational RAG chain"""
        question = inputs["question"]
        mode = inputs.get("mode", "general")  # general or examples
        
        if mode == "examples" and examples_retriever:
            # Retrieve example-focused documents
            example_docs = examples_retriever.get_relevant_documents(question)
            examples_context = "\n\n".join([doc.page_content for doc in example_docs])
            
            # Format the examples-focused prompt
            full_prompt = examples_prompt.format(
                examples_context=examples_context,
                question=question
            )
            
            source_docs = example_docs
            
        else:
            # Standard retrieval for general questions
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Get chat history
            chat_history = memory.format_for_prompt()
            
            # Format the prompt
            full_prompt = system_prompt.format(
                chat_history=chat_history,
                context=context,
                question=question
            )
            
            source_docs = docs
        
        # Get response from LLM
        try:
            # Try newer message format first
            messages = [{"role": "user", "content": full_prompt}]
            response = llm.invoke(messages)
            answer = response.content
        except:
            # Fallback to older format
            try:
                response = llm.invoke(full_prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
            except:
                # Last resort - direct call
                answer = llm(full_prompt)
        
        # Store in memory only for general mode
        if mode == "general":
            memory.add_exchange(question, answer)
        
        return {
            "answer": answer,
            "source_documents": source_docs,
            "chat_history": memory.format_for_prompt() if mode == "general" else "",
            "mode": mode
        }
    
    return conversational_rag_chain

# Streamlit app title
st.title("🎓 AI Study Assistant")
st.caption("With dedicated examples section for numerical problems!")

# Load API key securely from Streamlit Secrets
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("🚨 OPENAI_API_KEY not found. Please set it in Streamlit Secrets.")
    st.stop()

# Initialize session state variables
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "examples_history" not in st.session_state:
    st.session_state.examples_history = []

if "memory_messages" not in st.session_state:
    st.session_state.memory_messages = []

if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

if "examples_retriever" not in st.session_state:
    st.session_state.examples_retriever = None

if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# File upload
uploaded_file = st.file_uploader("Upload a PDF or Excel file", type=["pdf", "xlsx"])

# Check for a new file upload
if uploaded_file:
    if uploaded_file != st.session_state.uploaded_file:
        # Reset session state for a new file
        st.session_state.uploaded_file = uploaded_file
        st.session_state.chat_history = []
        st.session_state.examples_history = []
        if "conversation_memory" in st.session_state:
            st.session_state.conversation_memory = []  # Clear memory for new document
        st.session_state.retrieval_chain = None
        st.session_state.examples_retriever = None
        st.session_state.current_query = ""
        st.write("Processing new file...")

        # Process the uploaded file
        if uploaded_file.type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Use enhanced PDF extraction
            docs = extract_pdf_text(temp_file_path, method="auto")
            
            if not docs:
                st.error("❌ Failed to extract readable text from the PDF. The file might be:")
                st.write("- Image-based (scanned) PDF without OCR")
                st.write("- Heavily encrypted or protected")
                st.write("- Corrupted or in an unsupported format")
                st.write("Please try converting the PDF to a different format or use OCR.")
                st.stop()
            
            # Remove empty documents
            docs = [doc for doc in docs if doc.page_content.strip()]
            
            if not docs:
                st.error("❌ No readable text found in the PDF.")
                st.stop()
                
            # Show content preview
            content = docs[0].page_content[:1000]  # Show more content for preview
            st.write(f"### Document preview (first 1000 characters):")
            st.text_area("Extracted text:", content, height=200, disabled=True)
            
            # Show extraction statistics
            total_pages = len(docs)
            total_chars = sum(len(doc.page_content) for doc in docs)
            
            # Count example-containing pages
            example_pages = sum(1 for doc in docs if detect_problem_indicators(doc.page_content))
            
            st.info(f"📊 Successfully processed {total_pages} pages with {total_chars:,} characters")
            st.info(f"🔢 Found {example_pages} pages containing examples/problems")

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
            combined_text = df.to_string(index=False)
            docs = [Document(page_content=combined_text, metadata={"source": uploaded_file.name})]
            content = docs[0].page_content[:1000]
            st.write(f"### Document preview (first 1000 characters):")
            st.text_area("Excel content:", content, height=200, disabled=True)
        else:
            st.error("Unsupported file type. Please upload a PDF or Excel file.")
            st.stop()

        # Preprocess and split text with better chunking for LaTeX documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks for academic documents
            chunk_overlap=300,  # More overlap to preserve context
            separators=["\n\n", "\n", ". ", " ", ""]  # Better separators for academic text
        )
        splits = text_splitter.split_documents(docs)
        
        st.info(f"📄 Created {len(splits)} text chunks for processing")

        # Create vector store
        try:
            # Initialize embeddings first
            embeddings = OpenAIEmbeddings()
            
            # Create main vector store
            try:
                vectorstore = InMemoryVectorStore.from_documents(
                    documents=splits, 
                    embedding=embeddings
                )
            except Exception as e:
                # Fallback for older versions
                st.warning(f"InMemoryVectorStore failed, trying FAISS: {e}")
                from langchain.vectorstores import FAISS
                vectorstore = FAISS.from_documents(splits, embeddings)
            
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            # Create examples retriever
            examples_retriever = create_examples_retriever(splits, embeddings)
            st.session_state.examples_retriever = examples_retriever
            
            if examples_retriever:
                st.success("✅ Created specialized examples retriever!")
            else:
                st.warning("⚠️ No numerical examples detected in this document")

            # Set up the LLM
            try:
                llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
            except Exception as e:
                st.warning(f"Failed to initialize gpt-4o, trying gpt-3.5-turbo: {e}")
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
            
            # Create conversational chain with memory and examples
            st.session_state.retrieval_chain = create_conversational_chain(
                retriever, llm, examples_retriever
            )
            
            st.success("✅ Document processed successfully! Ready for questions with examples & memory!")
            
        except Exception as e:
            st.error(f"❌ Error creating retrieval chain: {str(e)}")
            st.error("This might be due to LangChain version incompatibility. Please try:")
            st.code("pip install langchain==0.0.352 langchain-openai==0.0.5")
            st.stop()

# Query interface with tabs
if st.session_state.retrieval_chain:
    
    # Create tabs for different modes
    tab1, tab2 = st.tabs(["💬 General Q&A", "🔢 Examples & Problems"])
    
    with tab1:
        st.write("### General Questions & Conceptual Understanding")
        
        # Show conversation context if available
        if "conversation_memory" in st.session_state and st.session_state.conversation_memory:
            with st.expander("💭 Full Conversation Context", expanded=False):
                full_context = format_chat_history_for_prompt(st.session_state.conversation_memory)
                st.text_area("Complete Conversation History", full_context, height=300)
        
        query = st.text_input("Ask your question:", placeholder="I'll remember our conversation context...", key="general_query")
        submit_query = st.button("Submit Question", key="general_submit")

        if submit_query and query.strip():
            with st.spinner("Thinking with context..."):
                try:
                    # 1. General answer
                    result_general = st.session_state.retrieval_chain({"question": query, "mode": "general"})
                    answer_general = result_general.get("answer", "No answer available.")
                    contexts_general = result_general.get("source_documents", [])
                    
                    st.session_state.chat_history.append({
                        "question": query, 
                        "answer": answer_general
                    })
                    
                    st.success("Answer:")
                    st.write(answer_general)

                    # Show source pages for general mode
                    if uploaded_file and uploaded_file.type == "application/pdf" and contexts_general:
                        pages = [ctx.metadata.get("page") for ctx in contexts_general if "page" in ctx.metadata]
                        if pages:
                            st.write("#### Answer retrieved from pages:")
                            st.write(", ".join([f"Page {p}" for p in sorted(set(pages))]))

                    # 2. Auto-trigger examples mode
                    if st.session_state.examples_retriever:
                        result_examples = st.session_state.retrieval_chain({"question": query, "mode": "examples"})
                        answer_examples = result_examples.get("answer", "No examples found.")
                        contexts_examples = result_examples.get("source_documents", [])

                        st.session_state.examples_history.append({
                            "question": query, 
                            "answer": answer_examples
                        })

                        # Show in tab1 too (optional)
                        st.info("🔢 Related Examples & Problems:")
                        st.write(answer_examples)

                        if uploaded_file and uploaded_file.type == "application/pdf" and contexts_examples:
                            pages_ex = [ctx.metadata.get("page") for ctx in contexts_examples if "page" in ctx.metadata]
                            if pages_ex:
                                st.write("#### Examples found on pages:")
                                st.write(", ".join([f"Page {p}" for p in sorted(set(pages_ex))]))

                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")


    
    with tab2:
        st.write("### Numerical Problems & Practical Examples")
        
        if not st.session_state.examples_retriever:
            st.warning("⚠️ No numerical examples detected in this document")
            st.write("This section works best with textbooks containing:")
            st.write("- Numbered examples or problems")
            st.write("- Step-by-step solutions")
            st.write("- Numerical calculations")
            st.write("- Practice exercises")
        else:
            st.info("🎯 This mode focuses specifically on finding numerical problems and practical examples from your document")
        
        examples_query = st.text_input(
            "Ask for examples or problems:", 
            placeholder="Show me examples of... / Find problems about... / Give me practice questions on...", 
            key="examples_query"
        )
        submit_examples = st.button("Find Examples", key="examples_submit")
        
        if submit_examples and examples_query.strip():
            with st.spinner("Searching for examples and problems..."):
                try:
                    result = st.session_state.retrieval_chain({
                        "question": examples_query, 
                        "mode": "examples"
                    })
                    
                    answer = result.get("answer", "No examples found.")
                    contexts = result.get("source_documents", [])
                    
                    st.session_state.examples_history.append({
                        "question": examples_query, 
                        "answer": answer
                    })
                    
                    st.success("📚 Examples & Problems Found:")
                    st.write(answer)
                    
                    # Show source pages for examples
                    if uploaded_file and uploaded_file.type == "application/pdf" and contexts:
                        pages = []
                        for context in contexts:
                            metadata = getattr(context, 'metadata', {})
                            if 'page' in metadata:
                                pages.append(metadata['page'])
                        
                        if pages:
                            unique_pages = sorted(set(pages))
                            st.write("#### Examples found on pages:")
                            st.write(", ".join([f"Page {page}" for page in unique_pages]))
                            
                            # Show example content preview
                            with st.expander("📖 Preview Example Content"):
                                for context in contexts[:2]:  # Show first 2 contexts
                                    page_num = getattr(context, 'metadata', {}).get('page', 'Unknown')
                                    content = getattr(context, 'page_content', '')[:800]
                                    st.write(f"**Page {page_num}:**")
                                    st.text_area(f"Content from page {page_num}", content, height=150, key=f"preview_{page_num}")
                                
                except Exception as e:
                    st.error(f"❌ Error finding examples: {str(e)}")

# Display chat histories
col1, col2 = st.columns(2)

with col1:
    if st.session_state.chat_history:
        st.write("### 💬 General Q&A History")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            with st.expander(f"Q{len(st.session_state.chat_history) - i}: {chat['question'][:50]}..."):
                st.write(f"**Question:** {chat['question']}")
                st.write(f"**Answer:** {chat['answer']}")

with col2:
    if st.session_state.examples_history:
        st.write("### 🔢 Examples History")
        for i, example in enumerate(reversed(st.session_state.examples_history[-5:])):  # Show last 5
            with st.expander(f"Ex{len(st.session_state.examples_history) - i}: {example['question'][:50]}..."):
                st.write(f"**Query:** {example['question']}")
                st.write(f"**Examples:** {example['answer']}")

# Memory visualization - show all conversation history
if "conversation_memory" in st.session_state and st.session_state.conversation_memory and st.checkbox("Show Full Conversation History"):
    st.write("### 💬 Complete Conversation History")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Messages", len(st.session_state.conversation_memory))
    with col2:
        st.metric("Conversation Turns", len(st.session_state.conversation_memory) // 2)
    
    # Show ALL memory as a timeline
    st.write("**Complete Conversation Timeline:**")
    for i, msg in enumerate(st.session_state.conversation_memory):
        role = "🧑‍🎓 Student" if msg['type'] == 'human' else "🤖 Tutor"
        with st.container():
            st.write(f"**{i+1}.** {role}: {msg['content'][:400]}...")
            st.markdown("---")

# Clear memory button
if st.sidebar.button("🗑️ Clear All Memory"):
    if "conversation_memory" in st.session_state:
        st.session_state.conversation_memory = []
    st.session_state.chat_history = []
    st.session_state.examples_history = []
    st.success("✅ All conversation memory cleared!")
