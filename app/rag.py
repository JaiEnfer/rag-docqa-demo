from pathlib import Path
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from .settings import settings
from .prompts import SYSTEM_PROMPT


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def load_or_create_store():
    index_path = Path(settings.vector_store_path)
    embeddings = get_embeddings()

    # FAISS saves as a folder; check if it exists
    if index_path.exists():
        return FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Create a new store with a dummy text (so FAISS has an index)
    store = FAISS.from_texts(["RAG store initialized."], embeddings)
    store.save_local(settings.vector_store_path)
    return store


def add_documents(store, docs):
    store.add_documents(docs)
    store.save_local(settings.vector_store_path)


def retrieve(store, query: str, k: int):
    return store.similarity_search(query, k=k)


def answer_with_ollama(question: str, context_docs):
    context = "\n\n".join(
        [f"[source={d.metadata.get('source', 'unknown')}] {d.page_content}" for d in context_docs]
    )

    prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    url = settings.ollama_url.rstrip("/") + "/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()
