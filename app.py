import streamlit as st
import fitz  # PyMuPDF
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Initialize the ChatGroq model
llm = ChatGroq(
    temperature=0,
    groq_api_key='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',
    model_name="llama-3.1-70b-versatile"
)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    doc = fitz.open(pdf_file_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Function to process the extracted text and question with ChatGroq
def process_text_with_llm(extracted_text, question):
    prompt_template = PromptTemplate.from_template(
        """
        ### EXTRACTED TEXT:
        {extracted_text}

        ### QUESTION:
        {question}

        ### INSTRUCTION:
        Based on the extracted text above, please provide a detailed and accurate answer to the question.
        ### ANSWER:
        """
    )
    chain = prompt_template | llm
    response = chain.invoke(input={'extracted_text': extracted_text, 'question': question})

    return response.content

# Streamlit app interface with chatbot-like interaction
st.title("Conversational PDF Q&A Bot")

# Initialize session state to store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File {uploaded_file.name} uploaded successfully!")

    # Extract text from the uploaded PDF
    extracted_text = extract_text_from_pdf("temp.pdf")

    # Display conversation
    st.write("---")
    st.markdown("<h3>Conversation</h3>", unsafe_allow_html=True)
    
    # Render conversation using st.chat_message
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input at the bottom
    if prompt := st.chat_input("Ask a question based on the PDF:"):
        # Add user's question to the conversation history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user's question in the chat interface
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get the response from the LLM
        with st.chat_message("assistant"):
            response = process_text_with_llm(extracted_text, prompt)
            st.markdown(response)
        
        # Add bot's response to the conversation history
        st.session_state.messages.append({"role": "assistant", "content": response})
