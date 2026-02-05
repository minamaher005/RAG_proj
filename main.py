

from config import validate_config
from document_loader import DocumentLoader
from embeddings import EmbeddingManager
from vector_store import VectorStore
from rag_pipeline import RAGPipeline


def initialize_rag_system():
    """Initialize all RAG components."""
    
    validate_config()
    
    
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        embedding_manager=embedding_manager
    )
    
    return embedding_manager, vector_store, rag_pipeline


def load_and_index_documents(
    embedding_manager: EmbeddingManager,
    vector_store: VectorStore
):
   
    doc_loader = DocumentLoader()
    texts = doc_loader.load_and_split()
    
    
    doc_content = DocumentLoader.get_content_list(texts)
    
    
    embeddings = embedding_manager.generate_embeddings(doc_content)
    vector_store.add_documents(texts, embeddings)
    
    print(f"Indexed {len(texts)} document chunks")
    return texts


def main():

    # Initialize system
    embedding_manager, vector_store, rag_pipeline = initialize_rag_system()
    
    while True:
        query = input("\nYour question: ").strip()
        
        if not query:
            continue
        

        results = rag_pipeline.query(query)
        
        print("\n Retrieved Context")
        print(results["context"])
        


if __name__ == "__main__":
    main()
