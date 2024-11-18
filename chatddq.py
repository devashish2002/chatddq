import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, NLTKTextSplitter
from langchain.schema import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
import os
import tempfile
import nltk

nltk.download('punkt_tab')

# Streamlit app title
st.title("AI Chatbot for financial Q/A")

api = st.secrets["api"]['openai_key']
os.environ["OPENAI_API_KEY"] = api

# Initialize session state variables
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

if "current_query" not in st.session_state:
    st.session_state.current_query = ""  # To store the current input in the query box

# File upload
uploaded_file = st.file_uploader("Upload a PDF or Excel file", type=["pdf", "xlsx"])

# Check for a new file upload
if uploaded_file:
    if uploaded_file != st.session_state.uploaded_file:
        # Reset session state for a new file
        st.session_state.uploaded_file = uploaded_file
        st.session_state.chat_history = []
        st.session_state.retrieval_chain = None
        st.session_state.current_query = ""  # Clear the query input field
        st.write("Processing new file...")

        # Process the uploaded file
        if uploaded_file.type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            content = docs[0].page_content[0:500]

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
            combined_text = df.to_string(index=False)
            docs = [Document(page_content=combined_text, metadata={"source": uploaded_file.name})]
            content = docs[0].page_content[0:500]
        else:
            st.error("Unsupported file type. Please upload a PDF or Excel file.")
            st.stop()

        st.write(f"### Document preview:")
        st.write(content)
        # Preprocess and split text
        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Create vector store
        vectorstore = InMemoryVectorStore.from_documents(
            documents=splits, embedding=OpenAIEmbeddings()
        )
        retriever = vectorstore.as_retriever()

        # Set up the LLM and retrieval chain
        llm = ChatOpenAI(model="gpt-4o")
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Keep the "
            "answer concise.\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        st.session_state.retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
    # else:
    #     st.write("File is already uploaded. Ready to ask questions!")

# Query input and response
if st.session_state.retrieval_chain:
    # Use a text input for the query but decouple execution with a button
    query =st.text_input("Ask your question:")
    submit_query = st.button("Submit Question")

    if submit_query: #and query.strip():
        # Update the query in session state and process it
        st.session_state.current_query = query
        with st.spinner("Fetching answer..."):
            result = st.session_state.retrieval_chain.invoke({"input": query})
            st.session_state.chat_history.append({"question": query, "answer": result.get("answer", "No answer available.")})
            st.success("Answer:")
            st.write(result.get("answer", "No answer available."))

            if uploaded_file.type == "application/pdf":
	            pages = [contexts.metadata['page'] for contexts in result['context']]
	            st.write(f"#### answer retreived from following pages")
	            for i, page in enumerate(pages, start=0):
	                st.write(f"Page {page}")

# Display chat history
if st.session_state.chat_history:
    st.write("### Chat History")
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        st.write(f"**Q{len(st.session_state.chat_history) - i}:** {chat['question']}")
        st.write(f"**A{len(st.session_state.chat_history) - i}:** {chat['answer']}")

# else:
#     st.info("Please upload a file to proceed.")
