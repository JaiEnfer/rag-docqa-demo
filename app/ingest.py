from pathlib import Path
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_pdf_file(path: Path) -> str:
    text = []
    with fitz.open(path) as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)

def chunk_text(text: str, source_name: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_text(text)
    return [Document(page_content=c, metadata={"source": source_name}) for c in chunks]
