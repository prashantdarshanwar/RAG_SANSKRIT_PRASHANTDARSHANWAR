import re
import logging

logger = logging.getLogger("SanskritRAG")


def clean_text(text):
    """
    Clean Sanskrit text for RAG processing.

    Steps:
    1. Remove extra spaces/newlines
    2. Remove unwanted special characters
    3. Preserve Sanskrit punctuation (। ॥) and Devanagari script
    """

    # [CHANGED] removed unidecode step — it converts Devanagari to Latin,
    # destroying original Sanskrit that the LLM and embedding model need.
    try:
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        if not text.strip():
            logger.warning("clean_text received empty or whitespace-only input.")
            return ""

        # Step 1: Collapse multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)

        # Step 2: Remove unwanted special characters, keep Sanskrit punctuation
        # and Devanagari unicode range (U+0900–U+097F)
        text = re.sub(r'[^\w\s।॥\u0900-\u097F]', '', text)

        # Step 3: Strip leading/trailing whitespace
        text = text.strip()

        logger.info(f"Text cleaned. Output length: {len(text)} characters.")
        return text

    except TypeError:
        raise

    except Exception as e:
        logger.error(f"Text cleaning failed: {e}")
        raise RuntimeError(f"clean_text error: {e}") from e


# Testing
if __name__ == "__main__":

    from src.utils.logger import setup_logger  # [CHANGED] fixed import path
    setup_logger()

    sample_text = """
    अथ योगानुशासनम् ॥

    योगश्चित्तवृत्तिनिरोधः ॥
    """

    try:
        cleaned = clean_text(sample_text)
        print("\n===== CLEANED TEXT =====\n")
        print(cleaned)

    except Exception as e:
        print(f"Error: {e}")