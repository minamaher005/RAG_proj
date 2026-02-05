"""Embedding generation using OpenAI."""

from typing import List
import numpy as np
from langchain_openai import OpenAIEmbeddings

from config import DEFAULT_EMBEDDING_MODEL


class EmbeddingManager:
    """Responsible for initializing and generating embeddings."""
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = OpenAIEmbeddings(model=self.model_name)
            print(f"Embedding model {self.model_name} loaded successfully")
        except Exception as e:
            print(f"Error loading the model {self.model_name}: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        try:
            print(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.embed_documents(texts)
            print(f"Generated embeddings for {len(embeddings)} texts")
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.model.embed_query(query)
