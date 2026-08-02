import os

import pytest
from dotenv import load_dotenv

load_dotenv()

# These smoke tests call the real OpenAI API and require an ingested vector DB
# (`python -m src.ingestion`), so they're skipped when no key is configured.
requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


@requires_openai
def test_stateless_chain_returns_grounded_answer():
    from src.chains.stateless_chain import get_farmer_bot_chain

    bot = get_farmer_bot_chain()
    response = bot.invoke("What subsidies are available for a paddy transplanter?")

    assert isinstance(response, str)
    assert len(response.strip()) > 0


@requires_openai
def test_conversational_chain_follows_up_with_history():
    from langchain_core.messages import AIMessage, HumanMessage

    from src.chains.conversational_chain import get_farmer_bot_chain

    bot = get_farmer_bot_chain()
    first_answer = bot({"input": "Tell me about the Seed Multiplication Scheme", "chat_history": []})
    assert len(first_answer.strip()) > 0

    history = [
        HumanMessage(content="Tell me about the Seed Multiplication Scheme"),
        AIMessage(content=first_answer),
    ]
    follow_up = bot({"input": "Who should I contact for it?", "chat_history": history})
    assert len(follow_up.strip()) > 0
