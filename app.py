import streamlit as st
import os, sys
sys.path.append(os.path.dirname(__file__))

from src.embedder import get_embeddings
from src.vector import build_vectorstore, load_vectorstore
from src.loader import load_documents
from src.chunker import chunk_documents
from src.reranker import rerank
from src.rag_chain import build_rag_chain, ask

st.set_page_config(page_title="🇹🇳 Tunisian Tax Assistant", layout="centered")
st.title("Tunisian Corporate Tax Assistant")
st.caption("Ask anything about Tunisian corporate tax law")

@st.cache_resource
def setup():
    embeddings = get_embeddings()
    if os.path.exists("./chroma_db"):
        vs = load_vectorstore(embeddings)
    else:
        docs = load_documents("./data")
        chunks = chunk_documents(docs)
        vs = build_vectorstore(chunks, embeddings)
    chain = build_rag_chain()
    return vs, chain

vectorstore, chain = setup()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if query := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer = ask(chain, vectorstore, rerank, query)
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})