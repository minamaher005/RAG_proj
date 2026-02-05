

from typing import List, Dict, Any

from embeddings import EmbeddingManager
from vector_store import VectorStore
from config import TOP_K_RESULTS


class RAGPipeline:
    
    def __init__(
        self, 
        vector_store: VectorStore, 
        embedding_manager: EmbeddingManager, 
        top_k: int = TOP_K_RESULTS
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.top_k = top_k
    
    def retrieve(self, query: str) -> Dict[str, Any]:
        
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_manager.generate_embeddings([query])[0]
            
            # Retrieve relevant documents
            results = self.vector_store.query(
                query_embedding=query_embedding,
                n_results=self.top_k
            )
            
            print(f"Retrieved {len(results['documents'][0])} documents for query: '{query}'")
            return results
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    
    def query(self, question: str) -> Dict[str, Any]:

        results = self.retrieve(question)
        context = self.format_context(results)
        
        return {
            "question": question,
            "context": context,
            "documents": results.get("documents", [[]])[0],
            "metadatas": results.get("metadatas", [[]])[0]
        }
