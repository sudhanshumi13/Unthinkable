from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from scripts.document_loader import load_document
import streamlit as st
from dotenv import load_dotenv
import os


load_dotenv()
st.set_page_config(page_title="Query-Bot", layout="wide")
st.title("LLM & RAG -Powered Document & Text Q&A")
st.markdown("""
Upload a **PDF** or **TXT** file, or paste large text below.  
Then ask a question and get instant, context-aware answers using LLM model
""")

uploaded_file = st.file_uploader("Upload PDF or Text", type=["pdf", "txt"])
large_text_input = st.text_area("Or paste your text here:", height=200)


chunks = []

if uploaded_file:
    temp_file = "./temp_input_file"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getvalue())
    chunks = load_document(temp_file)

elif large_text_input:
    doc = Document(page_content=large_text_input)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents([doc])


if not chunks:
    st.warning("Please upload a file or paste some text to continue.")
    st.stop()

st.info("Processing document/text... ⏳ Please wait a moment.")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("API_KEY")
)


system_prompt = (
    "You are a helpful AI assistant. Use the provided context to answer questions clearly and concisely. "
    "If the answer is not available in the context, say 'I don't know.'\n\nContext:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)
st.markdown("###Ask a Question About the Document/Text")
question = st.text_input("Type your question below:")
if question:
    with st.spinner("🤔 Thinking..."):
        response = chain.invoke({"input": question})['answer']
    st.markdown("### Answer is here:")
    st.success(response)


st.markdown("""
---
Made by Utkarsh Goel
""")
