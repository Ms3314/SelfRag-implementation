from dotenv import load_dotenv

load_dotenv()

DOCUMENTS_DIR = "./documents"
PERSIST_DIR = "./chroma_langchain_db"
COLLECTION_NAME = "example_collection"

CHAT_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "text-embedding-3-small"

MAX_RETRIES = 10
MAX_REWRITE_TRIES = 3
RETRIEVAL_K = 4
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
