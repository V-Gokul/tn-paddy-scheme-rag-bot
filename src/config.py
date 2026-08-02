"""Shared configuration for both the stateless and conversational RAG chains."""
import os

CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma_db")
DATA_PATH = os.getenv("DATA_PATH", "data/processed/tn_farmer_schemes.md")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "8"))
