from openai import OpenAI
import pandas as pd
import streamlit as st
import pdfplumber
import tiktoken

api_key = st.secrets["api"]["openai_key"]

client = OpenAI(api_key = api_key)

# client = OpenAI(
#     api_key="",
#     base_url="https://generativelanguage.googleapis.com/v1beta/"
# )

# Summarize large sections of the document to reduce token count
def summarize_large_document(text, max_tokens=128000):
    """Summarizes the document in sections if token count exceeds the limit."""
    #encoder = tiktoken.get_encoding("cl100k_base")
    tokens = count_tokens(text)
    
    if tokens <= max_tokens:
        return text  # No summarization needed
    
    # Split text into sections of about 10,000 tokens each
    section_size = 10000
    sections = [text[i:i + section_size] for i in range(0, tokens, section_size)]
    summaries = []

    #print(len(sections))

    st.write("Document is too large. Summarizing the first three sections...")
    
    for i, section in enumerate(sections):
        if i < 3:  # Summarize only the first three sections
            summary_prompt = f"Summarize the following text:\n\n{section}"
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=2000  # Limit to 2000 tokens per summary response
                )
                summary = response.choices[0].message.content.strip()
                summaries.append(summary)
                st.write(f"Section {i + 1} summarized.")
            except Exception as e:
                st.write(f"Error summarizing section {i + 1}: {str(e)}")
        else:
            # Append remaining sections without summarizing
            summaries.append(section)
    
    # Combine all summaries into a single condensed document
    condensed_text = " ".join(summaries)
    return condensed_text

def ask_openai(question: str, context: str) -> str:
    """Query the OpenAI API using the updated method."""
    prompt = f"""
You are a financial expert assistant. You will answer questions based on the following company financial data:
{context}

Question: {question}

Answer the question concisely and accurately.
    """
    try:
        # Use the `create` method for the updated API
        response = client.chat.completions.create(
            model="gpt-4o-mini",#"gemini-1.5-flash"
            messages=[
                {"role": "system", "content": "You are a financial assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content#.choices[0].message.content
    #     return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error occurred: {str(e)}"


def get_context_from_data(financial_data: pd.DataFrame, relevant_fields: list = None) -> str:
    """Convert financial data into a text format."""
    if relevant_fields:
        filtered_data = financial_data[relevant_fields]
    else:
        filtered_data = financial_data
    return filtered_data.to_string(index=False)

# Define the function to extract text from PDF
def extract_text_from_pdf(file):
    """Extracts text from each page of a PDF file and returns it as a single string."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def count_tokens(text):
    """Count tokens in the text using tiktoken with the GPT-4 encoder."""
    encoder = tiktoken.get_encoding("cl100k_base")  # For GPT-4 model
    tokens = encoder.encode(text)
    return len(tokens)

# Initialize session state for chat history, file, and context if not already done
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "context" not in st.session_state:
    st.session_state.context = None

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# if "preview" not in st.session_state:
#     st.session_state.preview = None


# Streamlit app layout
st.title("Financial Data Chatbot")
st.write("Upload your financial data and ask questions.")

# File uploader for Excel files
uploaded_file = st.file_uploader("Upload a file with financial data", type=["xlsx", "pdf"])

# Only process the file if a new one is uploaded
if uploaded_file and uploaded_file != st.session_state.uploaded_file:
    st.session_state.uploaded_file = uploaded_file  # Save the uploaded file in session state

    # Reset chat history when a new file is uploaded
    st.session_state.chat_history = []
    if "question" in st.session_state:
        st.session_state.question = ""
    if "preview" in st.session_state:
        st.session_state.preview = ""

    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        # Read the Excel file and generate context
        df = pd.read_excel(uploaded_file)
        st.write("### Financial Data Preview")
        st.dataframe(df)
        #st.session_state.preview = st.dataframe(df)
        context = get_context_from_data(df)
        
        st.session_state.preview = df.head(5).to_string(index=False)

    elif uploaded_file.type == "application/pdf":
        # Extract text from the PDF and generate context
        context = extract_text_from_pdf(uploaded_file)
        
        # Preview only the first 50 characters of the PDF content
        preview_text = context[:500] + "..."
        st.write("### PDF Content Preview")
        st.write(preview_text)
        
        st.session_state.preview = preview_text

    token_count = count_tokens(context)
    st.write(f"### Document Token Count: {token_count}")

    # Summarize if token count exceeds 128,000
    if token_count > 128000:
        st.write("The document is too large. Reducing size...")
        with st.spinner("Summarizing document..."):
            context = summarize_large_document(context)
        # token_count = count_tokens(context)
        # st.write(f"### Summarized Document Token Count: {token_count}")
        preview_text = ' '.join(context.split()[:100]) + "..."
        st.write("### Document preview")
        st.write(preview_text)

    st.session_state.context = context

if "preview" in st.session_state and st.session_state.preview:
    st.write("### File Preview")
    st.text(st.session_state.preview)
  
# If a file has been uploaded, display the question input and chat
if st.session_state.context:
    # Ask user to input a question
    question = st.text_input("Ask a question about your financial data:", key="question")

    # Process the question if it's entered
    if question:
        with st.spinner("Generating response..."):
            response = ask_openai(question, st.session_state.context)
            
            # Append the question and answer to the chat history
            st.session_state.chat_history.append({"question": question, "answer": response})
            
        # Display the chat history with the latest chat first
        st.write("### Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            st.write(f"**Q{len(st.session_state.chat_history) - i}:** {chat['question']}")
            st.write(f"**A{len(st.session_state.chat_history) - i}:** {chat['answer']}")
else:
    st.write("Please upload an Excel or PDF file to proceed.")  
