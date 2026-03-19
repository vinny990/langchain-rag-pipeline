import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHROMA_DIR = "chroma_db"

_qa_chain = None

PROMPT = ChatPromptTemplate.from_template(
    "You are an HR assistant. Answer the question using only the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def _get_chain():
    global _qa_chain
    if _qa_chain is None:
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        _qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | PROMPT
            | llm
            | StrOutputParser()
        )
    return _qa_chain


def ask(question: str) -> str:
    return _get_chain().invoke(question)
