from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

conversation_history = []

def build_rag_chain():
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2)

    prompt = PromptTemplate.from_template("""
You are a specialized assistant in Tunisian corporate tax law.
Answer questions about IS, VAT, tax declarations, incentives, and fiscal obligations.
Answer using ONLY the context below.
If not found, say "This information is not available in the provided documents."
Always cite the relevant article or law when possible.

Conversation history:
{history}

Context:
{context}

Question: {question}

Answer:
""")

    chain = prompt | llm | StrOutputParser()
    return chain

def ask(chain, vectorstore, rerank_fn, query):
    raw_chunks = vectorstore.similarity_search(query, k=6)
    best_chunks = rerank_fn(query, raw_chunks, top_n=3)
    context = "\n\n".join([c.page_content for c in best_chunks])
    
    history = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in conversation_history[-5:]])
    
    answer = chain.invoke({"context": context, "question": query, "history": history})
    
    conversation_history.append((query, answer))
    return answer