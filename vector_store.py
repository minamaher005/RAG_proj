"""Vector store management using ChromaDB."""

import os
import uuid
from typing import List, Any
import numpy as np
import chromadb

from config import VECTOR_STORE_PATH, COLLECTION_NAME


class VectorStore:
    """Manages the ChromaDB vector store for document storage and retrieval."""
    
    def __init__(
        self, 
        collection_name: str = COLLECTION_NAME,
        persist_directory: str = VECTOR_STORE_PATH
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize the ChromaDB client and collection."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            print(f"Vector store initialized. Collection '{self.collection_name}' has {self.collection.count()} documents")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Add documents and their embeddings to the vector store."""
        try:
            # Extract text content if documents are Document objects
            if hasattr(documents[0], 'page_content'):
                doc_texts = [doc.page_content for doc in documents]
            else:
                doc_texts = documents
            
            ids = [str(uuid.uuid4()) for _ in range(len(doc_texts))]
            metadatas = [{"source": f"document_{i}"} for i in range(len(doc_texts))]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=doc_texts
            )
            print(f"Added {len(doc_texts)} documents to the vector store")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
    
    def query(self, query_embedding: List[float], n_results: int = 5):
        """Query the vector store for similar documents."""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
    
    def get_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        print(f"Collection '{self.collection_name}' cleared")
