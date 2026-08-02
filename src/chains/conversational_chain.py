"""Multi-turn RAG chain: rephrases follow-up questions using chat history before
retrieval, so the bot understands references like "it" or "that scheme".
Compare with chains/stateless_chain.py.
"""
import logging

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    # ---------------------------------------------------------
    # Step 1: Rephrase Question Chain (Contextualizer)
    # ---------------------------------------------------------
    rephrase_system_prompt = (
        "Given a chat history and the latest user question which might reference context "
        "in the chat history, formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it if "
        "needed and otherwise return it as is."
    )

    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", rephrase_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    rephrase_chain = rephrase_prompt | llm | StrOutputParser()

    # ---------------------------------------------------------
    # Step 2: Main QA Prompt
    # ---------------------------------------------------------
    qa_system_prompt = (
        "You are an expert AI assistant providing accurate information on Tamil Nadu Government Agriculture & Paddy Schemes.\n"
        "Use the retrieved context below to answer the user's question.\n\n"
        "GUIDELINES:\n"
        "1. If the user asks for a general list of schemes or categories, summarize and list all scheme names mentioned in the context.\n"
        "2. If the user asks about a specific scheme, structure your response using:\n"
        "   - **Scheme Name & Objectives**\n"
        "   - **Benefits / Subsidy Offered**\n"
        "   - **Eligibility Criteria**\n"
        "   - **Officer to Contact**\n"
        "3. Only use facts directly present in the context. Do not invent details.\n\n"
        "Retrieved Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_chain = qa_prompt | llm | StrOutputParser()

    # ---------------------------------------------------------
    # Step 3: Conversational Pipeline Function
    # ---------------------------------------------------------
    def run_conversational_rag(inputs: dict) -> str:
        """
        inputs expected:
        {
            "input": "User's current question",
            "chat_history": [HumanMessage(...), AIMessage(...)]
        }
        """
        user_input = inputs["input"]
        chat_history = inputs.get("chat_history", [])

        if chat_history:
            standalone_query = rephrase_chain.invoke({
                "chat_history": chat_history,
                "input": user_input,
            })
        else:
            standalone_query = user_input

        docs = retriever.invoke(standalone_query)
        formatted_context = format_docs(docs)

        return qa_chain.invoke({
            "context": formatted_context,
            "chat_history": chat_history,
            "input": user_input,
        })

    return run_conversational_rag


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, AIMessage

    logging.basicConfig(level=logging.INFO)
    # print("🤖 Testing Conversational RAG...")
    logger.info("Testing conversational RAG...")
    bot = get_farmer_bot_chain()

    # print("\nTurn 1: Tell me about Seed Multiplication Scheme")
    res1 = bot({"input": "Tell me about Seed Multiplication Scheme of Paddy", "chat_history": []})
    print(res1)

    history = [
        HumanMessage(content="Tell me about Seed Multiplication Scheme of Paddy"),
        AIMessage(content=res1),
    ]
    # print("\nTurn 2 (Follow-up): Who should I contact for it?")
    res2 = bot({"input": "Who should I contact for it?", "chat_history": history})
    print(res2)
