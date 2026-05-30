import os

# BASE PROJECT PATHS
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# DATA PATH
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Sanskrit_doc.pdf")

FAISS_INDEX_PATH = os.path.join(BASE_DIR, "artifacts", "faiss_index")

LOG_DIR = os.path.join(BASE_DIR, "artifacts", "logs")

# EMBEDDING MODEL
# paraphrase-multilingual-MiniLM-L12-v2:
#   - Handles Devanagari natively — no transliteration needed
#   - 3x faster than mpnet, good quality for Sanskrit retrieval
EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# LLM — TinyLlama Chat uses Zephyr format: <|system|> <|user|> <|assistant|>
# CPU inference: ~15–40s per query depending on MAX_NEW_TOKENS
# To speed up: lower MAX_NEW_TOKENS to 150, or run on GPU (DEVICE=0)
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 200        # lower to 100–150 for faster CPU responses
TEMPERATURE    = 0.7        # unused when do_sample=False; kept for reference

# CHUNKING
CHUNK_SIZE    = 100
CHUNK_OVERLAP = 20

# RETRIEVAL
TOP_K_RESULTS = 3

# STREAMLIT
APP_TITLE = "Sanskrit RAG System"
APP_ICON  = "📚"

# DEVICE: -1 = CPU, 0 = GPU (cuda:0)
DEVICE = -1

# LOGGING
LOG_LEVEL = "INFO"