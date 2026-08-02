import logging
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from src.config import CHROMA_PATH, DATA_PATH, EMBEDDING_MODEL

load_dotenv()

logger = logging.getLogger(__name__)


def build_vector_db():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run 'python -m src.scraper' first!")

    # print(f"📖 Reading cleaned markdown from: {DATA_PATH}")
    logger.info("Reading cleaned markdown from: %s", DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Create a LangChain Document object directly
    documents = [Document(page_content=content, metadata={"source": DATA_PATH})]

    # Step 1: Split document into chunks
    # We split by headers and section dividers ('=' * 50 from our scraper)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n==================================================\n", "\n\n", "\n", " "],
    )

    chunks = text_splitter.split_documents(documents)
    # print(f" Total text chunks created: {len(chunks)}")
    logger.info("Total text chunks created: %d", len(chunks))

    # Step 2: Generate Embeddings and Save to ChromaDB
    # print(" Converting text chunks into Embeddings & storing in ChromaDB...")
    logger.info("Converting text chunks into embeddings & storing in ChromaDB...")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    # print(f" Vector Database successfully created and saved at: {CHROMA_PATH}")
    logger.info("Vector database successfully created and saved at: %s", CHROMA_PATH)
    return vector_db


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build_vector_db()
