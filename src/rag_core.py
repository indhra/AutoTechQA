import os
import dotenv

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config

dotenv.load_dotenv()

os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI(temperature=0, model=config.llm_model_name)

embedding_model_name = config.embedding_model_name


vector_store = Chroma(
    persist_directory=config.VECTOR_STORE_FOLDER,
    embedding_function=embedding_model_name)

retriever = vector_store.as_retriever(search_kwargs={'k': config.TOP_K_RETRIEVER})


# Chat prompt template
from langchain_core.prompts import ChatPromptTemplate

prompt_string = """
You are an expert assistant for automotive battery R&D.
Answer the user's question based *only* on the following context.
If the answer is not found in the context, say "I could not find an answer in the provided manuals."

Context:
{context}

Question:
{question}
"""
chat_prompt = ChatPromptTemplate.from_template(prompt_string)

## building the RAG Chain using LCEL
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_rag_chain():
    llm = chat_model
    retriever_final = retriever
    prompt = chat_prompt 
    
    # the LCEL chain
    rag_chain = (
        {"context":retriever_final, "question":RunnablePassthrough()}  
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

