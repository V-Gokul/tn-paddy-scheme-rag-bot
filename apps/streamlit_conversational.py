import logging
import sys
from pathlib import Path

# Running via `streamlit run apps/...` only puts this file's own folder on
# sys.path, not the project root — add the root so `src.*` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.chains.conversational_chain import get_farmer_bot_chain

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="TN Farmer Scheme AI Assistant", page_icon="🌾", layout="wide")


@st.cache_resource
def init_rag_chain():
    return get_farmer_bot_chain()


try:
    farmer_bot = init_rag_chain()
except Exception:
    logger.exception("Failed to load chain")
    st.error("Failed to load the assistant. Did you run `python -m src.ingestion`?")
    st.stop()

with st.sidebar:
    st.title("🌾 About the Portal")
    st.markdown("Provides information on **Tamil Nadu Paddy & Agriculture Schemes**.")
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.title("🌾 Tamil Nadu Farmer Scheme Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Vanakkam! 🌾 I am your AI assistant for Tamil Nadu Farmer Schemes. Ask me anything!",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask about schemes, subsidies, or contacts..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    langchain_history = []
    for msg in st.session_state.messages[1:-1]:
        if msg["role"] == "user":
            langchain_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_history.append(AIMessage(content=msg["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = farmer_bot({
                    "input": user_prompt,
                    "chat_history": langchain_history,
                })
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception:
                logger.exception("Error answering user query: %s", user_prompt)
                st.error("Something went wrong answering that question. Please try again.")
