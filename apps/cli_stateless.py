import logging
import sys
from pathlib import Path

# Running via `python apps/cli_stateless.py` only puts this file's own folder
# on sys.path, not the project root — add the root so `src.*` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.chains.stateless_chain import get_farmer_bot_chain

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("🌾 Welcome to the Tamil Nadu Farmer Scheme Assistant 🌾")
    print("Ask any question about paddy schemes, subsidies, or seeds.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("=" * 60)

    # Initialize the RAG chain once at startup
    print("\n⏳ Loading vector database and initializing AI chain...")
    try:
        farmer_bot = get_farmer_bot_chain()
        print(" Bot is ready!\n")
    except Exception:
        # print(f"❌ Error initializing bot: {e}")
        logger.exception("Error initializing bot")
        print("❌ Error initializing bot. Make sure you ran 'python -m src.ingestion' first.")
        sys.exit(1)

    # Interactive Chat Loop
    while True:
        try:
            user_input = input("\n🧑‍🌾 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 Thank you for using the Farmer Scheme Assistant. Vanakkam!")
                break

            print("\n🤖 Assistant is thinking...")
            response = farmer_bot.invoke(user_input)

            print("\n" + "-" * 60)
            print(response)
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\n👋 Session ended. Vanakkam!")
            break
        except Exception:
            # print(f"\n❌ An error occurred: {e}")
            logger.exception("Error handling user query")
            print("\n❌ Something went wrong answering that question. Please try again.")


if __name__ == "__main__":
    main()
