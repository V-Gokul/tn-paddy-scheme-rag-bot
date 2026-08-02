import logging
import os

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

from src.config import DATA_PATH

URL = "https://agritech.tnau.ac.in/expert_system/paddy/Schemes.html"
OUTPUT_DIR = os.path.dirname(DATA_PATH)

logger = logging.getLogger(__name__)


def scrape_tnau_schemes():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # print(f"🚀 Starting Playwright to scrape: {URL}")
    logger.info("Starting Playwright to scrape: %s", URL)

    with sync_playwright() as p:
        # Launch headless Chromium browser
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the page and wait for DOM content to load
        page.goto(URL, wait_until="domcontentloaded")
        page.wait_for_timeout(2000)  # Ensure all dynamic elements settle

        # Extract fully rendered HTML content
        content = page.content()
        browser.close()

    # print("Parsing HTML content with BeautifulSoup...")
    logger.info("Parsing HTML content with BeautifulSoup...")
    soup = BeautifulSoup(content, "html.parser")

    # Remove irrelevant web elements (navigation, headers, script tags)
    for element in soup(["script", "style", "nav", "header", "footer"]):
        element.decompose()

    # Extract clean text and structure
    text_lines = []

    # Process tables specifically to maintain structural context for RAG
    tables = soup.find_all("table")
    if not tables:
        raise ValueError(
            f"No tables found on {URL}. The page layout may have changed; "
            "the scraper needs to be updated."
        )

    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cols = [col.get_text(strip=True) for col in row.find_all(["td", "th"])]
            if cols:
                # Format table row as pipe-separated markdown format
                text_lines.append(" | ".join(cols))
        text_lines.append("\n" + "=" * 50 + "\n")

    full_text = "\n".join(text_lines)

    # Save to Markdown (ideal for RAG parsing)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write("# Tamil Nadu Farmer Paddy Schemes\n\n")
        f.write(full_text)

    # print(f" Saved cleaned schemes data to: {md_path}")
    logger.info("Saved cleaned schemes data to: %s", DATA_PATH)
    return DATA_PATH


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    scrape_tnau_schemes()
