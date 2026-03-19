# langchain-rag-pipeline

An HR policy chatbot rebuilt with LangChain, replacing a manual RAG implementation.

**Live demo:** https://langchain-rag-pipeline.onrender.com

## What LangChain replaced

| Concern | Original (`hr-ai-assistant`) | This project |
|---|---|---|
| PDF loading | `PyPDF2.PdfReader` + manual page loop | `PyPDFLoader` |
| Chunking | Word-count splitter (fixed 150-word windows, no overlap) | `RecursiveCharacterTextSplitter` (character-aware, configurable overlap) |
| Embeddings | Direct `openai.embeddings.create` call, results stored as a NumPy array in memory | `OpenAIEmbeddings` via LangChain, persisted to disk in Chroma |
| Vector search | Manual cosine similarity with `np.dot` / `np.linalg.norm` | Chroma's built-in similarity search via LangChain `Retriever` |
| LLM call | Raw `openai.chat.completions.create` with a hand-built prompt string | `RetrievalQA` chain with `ChatOpenAI` — retrieval + prompt + parsing handled automatically |
| Persistence | None (index rebuilt from scratch on every startup) | Chroma persists to `chroma_db/`; ingest runs once |

## Setup

```bash
cp .env.example .env
# add your OPENAI_API_KEY to .env

pip install -r requirements.txt

# one-time: embed and persist the PDF
python ingest.py

# run the app
python app.py
```

Open http://localhost:5000 in your browser.

## Project structure

```
langchain-rag-pipeline/
├── uploads/          # source PDFs
├── chroma_db/        # persisted vector store (created by ingest.py)
├── ingest.py         # load → chunk → embed → persist
├── rag_chain.py      # load Chroma, build RetrievalQA, expose ask()
├── app.py            # Flask: GET / (chat UI), POST /ask
├── requirements.txt
└── .env.example
```
