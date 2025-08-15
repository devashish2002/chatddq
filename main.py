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
        from langchain.chains import RetrievalQA
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
import pandas as pd
import os
import tempfile
import nltk
import re
import unicodedata
import fitz  # PyMuPDF - better for complex PDFs
import pdfplumber  # Another good option for PDF parsing

nltk.download('punkt_tab')

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

# Streamlit app title
st.title("AI study assistant")

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

if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

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
        st.session_state.retrieval_chain = None
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
            st.info(f"📊 Successfully processed {total_pages} pages with {total_chars:,} characters")

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
            
            # Create vector store with better error handling
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
            
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve more relevant chunks

            # Set up the LLM and retrieval chain with better error handling
            try:
                llm = ChatOpenAI(model="gpt-4o", temperature=0)
            except Exception as e:
                st.warning(f"Failed to initialize gpt-4o, trying gpt-3.5-turbo: {e}")
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            
            system_prompt = (
                "You are an assistant for question-answering tasks, specializing in academic and financial documents. "
                "Use the following pieces of retrieved context to answer the question. "
                "If mathematical formulas or technical terms appear garbled in the context, try to infer their meaning from surrounding text. "
                "If you don't know the answer, say that you don't know. "
                "Keep the answer concise but comprehensive. "
                "Based on your answer, suggest 3 relevant follow-up questions the user might want to ask.\n\n"
                "{context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            # Try new chain creation method first, then fallback to older methods
            try:
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                st.session_state.retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
                st.session_state.chain_type = "new"
            except Exception as e:
                st.warning(f"New chain method failed, trying RetrievalQA: {e}")
                # Fallback to older RetrievalQA method
                from langchain.chains import RetrievalQA
                st.session_state.retrieval_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )
                st.session_state.chain_type = "old"
            
            st.success("✅ Document processed successfully! Ready for questions.")
            
        except Exception as e:
            st.error(f"❌ Error creating retrieval chain: {str(e)}")
            st.error("This might be due to LangChain version incompatibility. Please try:")
            st.code("pip install langchain==0.0.352 langchain-openai==0.0.5")
            st.stop()

# Query input and response
if st.session_state.retrieval_chain:
    query = st.text_input("Ask your question:")
    submit_query = st.button("Submit Question")

    if submit_query and query.strip():
        st.session_state.current_query = query
        with st.spinner("Fetching answer..."):
            try:
                # Handle different chain types with different input formats
                chain = st.session_state.retrieval_chain
                
                # Try different input formats based on chain type
                result = None
                answer = None
                contexts = []
                
                # Method 1: Try new retrieval chain format
                try:
                    result = chain.invoke({"input": query})
                    answer = result.get("answer", "No answer available.")
                    contexts = result.get("context", [])
                except Exception as e1:
                    # Method 2: Try RetrievalQA format
                    try:
                        result = chain({"query": query})
                        answer = result.get("result", "No answer available.")
                        contexts = result.get("source_documents", [])
                    except Exception as e2:
                        # Method 3: Try direct invocation with query
                        try:
                            result = chain.invoke({"query": query})
                            answer = result.get("answer", result.get("result", "No answer available."))
                            contexts = result.get("context", result.get("source_documents", []))
                        except Exception as e3:
                            # Method 4: Last resort - try run method
                            try:
                                answer = chain.run(query)
                                contexts = []
                            except Exception as e4:
                                raise Exception(f"All methods failed. Errors: {e1}, {e2}, {e3}, {e4}")
                
                st.session_state.chat_history.append({
                    "question": query, 
                    "answer": answer
                })
                
                st.success("Answer:")
                st.write(answer)

                # Show source pages for PDF files
                if uploaded_file and uploaded_file.type == "application/pdf" and contexts:
                    pages = []
                    for context in contexts:
                        # Handle both Document objects and dictionaries
                        metadata = None
                        if hasattr(context, 'metadata'):
                            metadata = context.metadata
                        elif isinstance(context, dict) and 'metadata' in context:
                            metadata = context['metadata']
                        
                        if metadata and 'page' in metadata:
                            pages.append(metadata['page'])
                    
                    if pages:
                        unique_pages = sorted(set(pages))
                        st.write("#### Answer retrieved from the following pages:")
                        for page in unique_pages:
                            st.write(f"Page {page}")
                            
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.error("Debug info: Please check if your LangChain installation is complete.")
                
                # Show debug information
                if st.checkbox("Show debug information"):
                    st.write("Chain type:", type(st.session_state.retrieval_chain))
                    st.write("Available methods:", [method for method in dir(st.session_state.retrieval_chain) if not method.startswith('_')])
                    st.write("Error details:", str(e))

# Display chat history
if st.session_state.chat_history:
    st.write("### Chat History")
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Q{len(st.session_state.chat_history) - i}: {chat['question'][:100]}..."):
            st.write(f"**Question:** {chat['question']}")
            st.write(f"**Answer:** {chat['answer']}")