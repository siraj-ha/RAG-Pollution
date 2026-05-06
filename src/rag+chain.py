from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def build_rag_chain():
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0.2
    )

    prompt = PromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:
""")

    chain = prompt | llm | StrOutputParser()
    return chain