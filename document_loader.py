

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.schema import Document

from config import PDF_DIRECTORY, CHUNK_SIZE, CHUNK_OVERLAP


class DocumentLoader:
   
    
    def __init__(
        self, 
        pdf_directory: str = PDF_DIRECTORY,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.pdf_directory = pdf_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def load_documents(self) -> List[Document]:
        """Load all PDF files from the directory."""
        print(f"Loading PDF files from {self.pdf_directory}")
        pdf_loader = DirectoryLoader(
            self.pdf_directory,
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=True
        )
        documents = pdf_loader.load()
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        texts = self.text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks")
        return texts
    
    def load_and_split(self) -> List[Document]:
        """Load and split documents in one step."""
        documents = self.load_documents()
        return self.split_documents(documents)
    
    @staticmethod
    def get_content_list(documents: List[Document]) -> List[str]:
        """Extract page content from documents."""
        return [doc.page_content for doc in documents]
