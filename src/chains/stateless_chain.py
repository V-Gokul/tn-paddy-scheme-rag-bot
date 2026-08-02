"""Single-turn RAG chain: each question is answered independently, with no memory
of prior turns. Compare with chains/conversational_chain.py.
"""
import logging

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.config import CHROMA_PATH, EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE, RETRIEVER_K

load_dotenv()

logger = logging.getLogger(__name__)


def format_docs(docs):
    """Combine retrieved documents into a single text block."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def get_farmer_bot_chain():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVER_K})

    system_prompt = (
        "You are an expert AI assistant providing accurate information on Tamil Nadu Government Agriculture & Paddy Schemes.\n"
        "Use ONLY the retrieved context provided below to answer the user's question.\n"
        "If you do not know the answer based on the context, state that you don't have enough details rather than making up facts.\n\n"
        "When answering, structure your response clearly using bullet points with:\n"
        "- **Scheme Name & Objectives**\n"
        "- **Benefits / Subsidy Offered**\n"
        "- **Eligibility Criteria**\n"
        "- **Officer to Contact**\n\n"
        "Retrieved Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # print("🤖 Initializing Tamil Nadu Farmer Assistant...")
    logger.info("Initializing Tamil Nadu Farmer Assistant (stateless)...")
    bot_chain = get_farmer_bot_chain()

    test_query = "What subsidies are available for purchasing a paddy transplanter or conoweeder?"
    # print(f"\n❓ Query: {test_query}\n")
    logger.info("Query: %s", test_query)

    response = bot_chain.invoke(test_query)
    # print("💡 Answer:")
    print(response)
