import streamlit as st
import os
from utils.pdf_reader import extract_text_from_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

@st.cache_resource
def load_documents():
    all_docs = []
    
    for file in os.listdir("data/"):
        if file.endswith('.pdf'):
            path = os.path.join("data", file)
            text = extract_text_from_pdf(path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.create_documents([text], metadatas=[{"source": file}])
            all_docs.extend(chunks)
    
    return all_docs

@st.cache_resource
def get_vector_store(_docs, index_path="faiss_index"):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings=embedding, allow_dangerous_deserialization=True)
    
    vectorstore = FAISS.from_documents(_docs, embedding)
    vectorstore.save_local(index_path)
    return vectorstore

@st.cache_resource
@st.cache_resource
def load_phi():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )

    phi_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    return phi_pipeline

@st.cache_resource
def get_qa_chain(vectorstore, phi_pipeline):
    llm = HuggingFacePipeline(pipeline=phi_pipeline)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

#Chatbot Interface
st.set_page_config(page_title="IIITDM Chatbot", layout="wide")
st.title("IIITDM Academic FAQ Chatbot")

query = st.text_input("Ask your question:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if query:
    docs = load_documents()
    vectorstore = get_vector_store(docs)
    phi_pipeline = load_phi()
    qa_chain = get_qa_chain(vectorstore, phi_pipeline)

    answer = qa_chain.run(query)
    st.session_state.chat_history.append((query, answer))

for q, a in st.session.chat_history[::-1]:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
