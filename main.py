import os
import sys
sys.path.append(os.path.dirname(__file__))

from src.loader import load_documents
from src.chunker import chunk_documents
from src.embedder import get_embeddings
from src.vector import build_vectorstore, load_vectorstore
from src.rag_chain import build_rag_chain

if __name__ == "__main__":
    embeddings = get_embeddings()

    if os.path.exists("./chroma_db"):
        print("Loading existing vector store...")
        vectorstore = load_vectorstore(embeddings)
    else:
        print("Building vector store for the first time...")
        docs = load_documents("./data")
        chunks = chunk_documents(docs)
        vectorstore = build_vectorstore(chunks, embeddings)

    chain = build_rag_chain()

    print("\nReady! Type your question (or 'quit' to exit)\n")
    while True:
        query = input("You: ").strip()
        if query.lower() == "quit":
            break

        results = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([r.page_content for r in results])
        answer = chain.invoke({"context": context, "question": query})
        print(f"\nAssistant: {answer}\n")