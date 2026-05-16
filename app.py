import streamlit as st
import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

from src.embedder import get_embeddings
from src.vector import build_vectorstore, load_vectorstore
from src.loader import load_documents
from src.chunker import chunk_documents
from src.reranker import rerank
from src.rag_chain import build_rag_chain, ask

# ─── PAGE CONFIG ───
st.set_page_config(
    page_title="Tunisian Tax Assistant",
    page_icon="🇹🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #E70013, #C0392B);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .stat-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin: 5px;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #E70013;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #666;
    }
    .source-card {
        background: #f0f7ff;
        border-left: 4px solid #2E75B6;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        font-size: 0.85rem;
        color: #222;
    }
    .log-entry {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        border-left: 3px solid #E70013;
        font-size: 0.85rem;
    }
    .user-msg {
        background: #E70013;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 5px 18px;
        margin: 5px 0;
        max-width: 80%;
        margin-left: auto;
    }
    .assistant-msg {
        background: #f0f0f0;
        color: #222;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 5px;
        margin: 5px 0;
        max-width: 85%;
    }
    .doc-badge {
        background: #E70013;
        color: white;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.75rem;
        margin: 2px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ─── INIT SESSION STATE ───
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_logs" not in st.session_state:
    st.session_state.query_logs = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "sources_used" not in st.session_state:
    st.session_state.sources_used = []

# ─── LOAD SYSTEM ───
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

# ─── GET DB STATS ───
def get_db_stats():
    try:
        collection = vectorstore._collection
        count = collection.count()
        return count
    except:
        return 0

def get_loaded_docs():
    try:
        data_path = "./data"
        if os.path.exists(data_path):
            files = [f for f in os.listdir(data_path) if f.endswith(".pdf")]
            return files
        return []
    except:
        return []

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("## 🇹🇳 Tax Assistant")
    st.markdown("---")

    # DB Stats
    st.markdown("### 📊 System Status")
    chunk_count = get_db_stats()
    loaded_docs = get_loaded_docs()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{chunk_count}</div>
            <div class='stat-label'>Chunks indexed</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{len(loaded_docs)}</div>
            <div class='stat-label'>Documents</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-number'>{st.session_state.total_queries}</div>
        <div class='stat-label'>Total queries this session</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Document Manager
    st.markdown("### 📁 Document Manager")
    if loaded_docs:
        for doc in loaded_docs:
            name = doc.replace("-", " ").replace("_", " ").replace(".pdf", "")
            st.markdown(f"<span class='doc-badge'>📄 {name[:30]}...</span>" if len(name) > 30 else f"<span class='doc-badge'>📄 {name}</span>", unsafe_allow_html=True)
    else:
        st.warning("No documents in /data folder")

    st.markdown("---")

    # ChromaDB Collection Info
    st.markdown("### 🗄️ ChromaDB Collection")
    try:
        col_name = vectorstore._collection.name
        st.info(f"**Collection:** {col_name}\n\n**Chunks:** {chunk_count}\n\n**Status:** ✅ Active")
    except:
        st.info("Collection active")

    st.markdown("---")

    # Controls
    st.markdown("### ⚙️ Controls")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources_used = []
        st.rerun()

    if st.button("📋 Export chat logs", use_container_width=True):
        if st.session_state.query_logs:
            log_text = "\n\n".join([
                f"[{log['time']}]\nQ: {log['question']}\nA: {log['answer']}\nSources: {', '.join(log['sources'])}"
                for log in st.session_state.query_logs
            ])
            st.download_button(
                "⬇️ Download logs",
                data=log_text,
                file_name=f"tax_assistant_logs_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.warning("No logs yet")

    # Suggested questions
    st.markdown("---")
    st.markdown("### 💡 Suggested Questions")
    suggestions = [
        "Quel est le taux de l'IS ?",
        "Quelles sont les exonérations TVA ?",
        "Délai de déclaration fiscale ?",
        "Pénalités retard fiscal ?",
        "Avantages entreprises exportatrices ?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=s):
            st.session_state["suggested_query"] = s

# ─── MAIN AREA ───
st.markdown("""
<div class='main-header'>
    <h2>🇹🇳 Tunisian Corporate Tax Law Assistant</h2>
    <p>Powered by RAG · ChromaDB · Llama 3.1 · LangChain</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📋 Query Logs", "📊 Analytics"])

# ─── TAB 1: CHAT ───
with tab1:
    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-msg'>🧑 {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-msg'>🤖 {msg['content']}</div>", unsafe_allow_html=True)
                if "sources" in msg and msg["sources"]:
                    with st.expander("📎 Sources used"):
                        for src in msg["sources"]:
                            st.markdown(f"<div class='source-card'>{src}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Handle suggested query from sidebar
    default_val = st.session_state.pop("suggested_query", "")

    query = st.chat_input("Ask your tax question in French or English...")

    if query or default_val:
        user_input = query or default_val

        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("🔍 Searching documents and generating answer..."):
            raw_chunks = vectorstore.similarity_search(user_input, k=6)
            best_chunks = rerank(user_input, raw_chunks, top_n=3)

            sources = []
            for chunk in best_chunks:
                meta = chunk.metadata
                source = meta.get("source", "Unknown")
                page = meta.get("page", "?")
                filename = os.path.basename(source)
                sources.append(f"📄 **{filename}** — Page {page}")

            answer = ask(chain, vectorstore, rerank, user_input)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

        st.session_state.total_queries += 1
        st.session_state.query_logs.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "question": user_input,
            "answer": answer,
            "sources": sources
        })

        st.rerun()

# ─── TAB 2: QUERY LOGS ───
with tab2:
    st.markdown("### 📋 Query History")
    if st.session_state.query_logs:
        for i, log in enumerate(reversed(st.session_state.query_logs)):
            with st.expander(f"[{log['time']}] {log['question'][:60]}..."):
                st.markdown(f"**❓ Question:** {log['question']}")
                st.markdown(f"**🤖 Answer:** {log['answer']}")
                st.markdown("**📎 Sources:**")
                for src in log["sources"]:
                    st.markdown(f"<div class='source-card'>{src}</div>", unsafe_allow_html=True)
    else:
        st.info("No queries yet. Start asking questions in the Chat tab.")

# ─── TAB 3: ANALYTICS ───
with tab3:
    st.markdown("### 📊 Session Analytics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Queries", st.session_state.total_queries)
    with col2:
        st.metric("Documents Loaded", len(loaded_docs))
    with col3:
        st.metric("Chunks in DB", chunk_count)

    st.markdown("---")
    st.markdown("### 🗄️ ChromaDB Details")

    try:
        col_info = {
            "Collection name": vectorstore._collection.name,
            "Total chunks": chunk_count,
            "Embedding model": "all-MiniLM-L6-v2",
            "Chunk size": "1000 characters",
            "Chunk overlap": "150 characters",
            "Retrieval k": "6 (re-ranked to top 3)",
        }
        for k, v in col_info.items():
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown(f"**{k}**")
            with col_b:
                st.markdown(str(v))
    except Exception as e:
        st.error(str(e))

    st.markdown("---")
    st.markdown("### 📁 Loaded Documents")
    for doc in loaded_docs:
        size = os.path.getsize(f"./data/{doc}") / 1024
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"📄 {doc}")
        with col_b:
            st.markdown(f"`{size:.1f} KB`")
