import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

UPLOADS_DIR = "uploads"
CHROMA_DIR = "chroma_db"


def ingest():
    pdf_files = [f for f in os.listdir(UPLOADS_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in uploads/")
        return

    docs = []
    for filename in pdf_files:
        path = os.path.join(UPLOADS_DIR, filename)
        print(f"Loading {filename}...")
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    print(f"Loaded {len(docs)} pages. Chunking...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks. Embedding and persisting...")

    embeddings = OpenAIEmbeddings()
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    print(f"Done. Vector store saved to {CHROMA_DIR}/")


if __name__ == "__main__":
    ingest()
