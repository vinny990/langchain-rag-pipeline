import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

CHROMA_DIR = "chroma_db"

_qa_chain = None


def _get_chain():
    global _qa_chain
    if _qa_chain is None:
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        _qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
        )
    return _qa_chain


def ask(question: str) -> str:
    chain = _get_chain()
    result = chain.invoke({"query": question})
    return result["result"]
