from langchain_community.vectorstores import Chroma

PERSIST_DIR = "./chroma_db"

def build_vectorstore(chunks, embeddings):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print("Vector store saved.")
    return vectorstore

def load_vectorstore(embeddings):
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )