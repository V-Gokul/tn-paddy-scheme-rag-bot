import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DATA_PATH


def test_scraped_data_file_exists():
    assert os.path.exists(DATA_PATH), (
        f"Missing {DATA_PATH}. Run 'python -m src.scraper' first."
    )


def test_scraped_data_produces_chunks():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    documents = [Document(page_content=content, metadata={"source": DATA_PATH})]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n==================================================\n", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(documents)

    assert len(chunks) > 0
    assert all(chunk.page_content.strip() for chunk in chunks)
