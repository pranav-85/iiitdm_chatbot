import streamlit as st
import os
import re
from utils.pdf_reader import extract_text_from_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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
def load_llama():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )

    llama_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    return llama_pipeline

@st.cache_resource
def get_qa_chain(_vectorstore, _llama_pipeline):
    # Define the LLM wrapper
    llm = HuggingFacePipeline(pipeline=_llama_pipeline)

    # Define your custom prompt template
    prompt_template = """
    You are a helpful assistant. Use the following context to answer the question as clearly and concisely as possible.

    Context:
    {context}

    Question:
    {question}

    Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    # Create the QA chain with custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=_vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True  # Optional: include source documents in the result
    )

    return qa_chain

#Chatbot Interface
st.set_page_config(page_title="IIITDM Chatbot", layout="wide")
st.title("IIITDM Academic FAQ Chatbot")

query = st.text_input("Ask your question:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def clean_response(text):
    """
    Generic cleaner to remove prompt template noise, context dumps, and leave a clean answer.
    """
    import re

    # Remove leading template phrases
    text = re.sub(r"(?i)^You are a helpful assistant.*?\n", "", text)

    # Remove everything from 'Context:' to 'Question:' or 'Answer:' (if present)
    text = re.sub(r"(?is)context:.*?(question:|answer:)", "", text)

    # Remove “Question:” and “Answer:” headers
    text = re.sub(r"(?i)question\s*:\s*", "", text)
    text = re.sub(r"(?i)answer\s*:\s*", "", text)

    # Remove redundant line breaks and whitespace
    text = re.sub(r"\n+", "\n", text).strip()

    return text

if query:
    docs = load_documents()
    vectorstore = get_vector_store(docs)
    llama_pipeline = load_llama()
    qa_chain = get_qa_chain(vectorstore, llama_pipeline)

    response = qa_chain.invoke({"query": query})
    raw_answer = response["result"]
    cleaned_answer = clean_response(raw_answer)
    sources = response["source_documents"]

    st.session_state.chat_history.append((query, cleaned_answer, sources))

# Chat history display
for q, a, s in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑 You:** {q}")
    st.markdown(f"**🤖 Bot:** {a}")
    with st.expander("📄 Source Documents"):
        for i, doc in enumerate(s):
            st.markdown(f"**Source {i+1}:**")
            st.write(doc.page_content)
