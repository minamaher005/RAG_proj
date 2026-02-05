"""Configuration and environment setup for the RAG application."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding Configuration
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"

# Vector Store Configuration
VECTOR_STORE_PATH = "./data/vector_store"
COLLECTION_NAME = "pdf-docs"

# Document Loading Configuration
PDF_DIRECTORY = "./data/pdf"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

# RAG Configuration
TOP_K_RESULTS = 5


def validate_config():
    """Validate that required configuration is present."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please add it to your .env file.")
    return True
