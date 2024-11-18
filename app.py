import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit title
st.title("Know Me!")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt
prompt = ChatPromptTemplate.from_template(
    """
    Identify as Rahul and answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Function to handle vector embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./dataset")  # Data ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Create vector store

# Display example questions
st.subheader("Example Questions:")
example_questions = [
    "Introduce Yourself.",
    "What are your technical skills?",
    "What projects have you worked on?",
    "Why do you think you will be a good fit for TCS Prime Role?"
]
clicked_question = st.radio("Click on a question to get an answer:", example_questions)

# Input for custom questions
prompt1 = st.text_input("Or enter your own question:")

# Load vector embeddings
vector_embedding()

# Handle question input (clicked example or user input)
question = prompt1 or clicked_question

if question:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Generate response
    start = time.process_time()
    response = retrieval_chain.invoke({'input': question})
    response_time = time.process_time() - start
    st.write(f"**Response:** {response['answer']}")
    st.write(f"*Response time: {response_time:.2f} seconds*")
