import logging
import sys
from pathlib import Path

# Running via `streamlit run apps/...` only puts this file's own folder on
# sys.path, not the project root — add the root so `src.*` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

from src.chains.stateless_chain import get_farmer_bot_chain

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
    logger.exception("Failed to load Vector DB")
    st.error("Failed to load the vector database. Did you run `python -m src.ingestion`?")
    st.stop()

with st.sidebar:
    st.title("🌾 About the Portal")
    st.markdown(
        """
        This AI Assistant provides accurate details on **Tamil Nadu Government Paddy & Agriculture Schemes**.

        **Data Source:**
        [TNAU Agritech Expert System](https://agritech.tnau.ac.in/expert_system/paddy/Schemes.html)

        **Key Features Covered:**
        - Seed Multiplication & Quality Seeds
        - Machinery Subsidies (Transplanters, Conoweeders)
        - Rainwater Harvesting & Irrigation
        - Contact Officers & Eligibility
        """
    )
    st.divider()
    st.caption("Powered by LangChain, OpenAI, and ChromaDB")

st.title("🌾 Tamil Nadu Farmer Scheme Assistant")
st.subheader("Ask questions about subsidies, seed schemes, and agricultural equipment support.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Vanakkam! 🌾 I am your AI assistant for Tamil Nadu Farmer Schemes. How can I assist you today?",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask about paddy schemes, subsidies, or contacts..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching scheme details..."):
            try:
                response = farmer_bot.invoke(user_prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception:
                logger.exception("Error answering user query: %s", user_prompt)
                st.error("Something went wrong answering that question. Please try again.")
