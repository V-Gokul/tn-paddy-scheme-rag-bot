#!/bin/sh
set -e

CHROMA_DB_FILE="${CHROMA_PATH:-data/chroma_db}/chroma.sqlite3"

if [ ! -f "$CHROMA_DB_FILE" ]; then
    echo "Vector DB not found at $CHROMA_DB_FILE - building it now..."
    python -m src.ingestion
fi

exec streamlit run apps/streamlit_stateless.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
