from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Global variable to store the vector store instance
_vector_store = None

def get_qdrant_store(collection_name="pdf_docs"):
    """
    Create or return existing FAISS vector store for local development.
    This eliminates the need for running Qdrant separately.
    """
    global _vector_store
    
    if _vector_store is not None:
        print("🔄 Using existing FAISS vector store")
        return _vector_store
    
    print("🔄 Creating new FAISS in-memory vector store for local development")
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    # Create an empty FAISS index with dummy text
    # This will be populated when documents are added
    try:
        _vector_store = FAISS.from_texts(["Welcome to the RAG Chat App"], embeddings)
        print("✅ FAISS vector store created successfully")
        return _vector_store
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        raise e

def reset_vector_store():
    """Reset the global vector store (useful for testing or new PDF uploads)"""
    global _vector_store
    _vector_store = None
    print("🔄 Vector store reset - ready for new content")

def create_fresh_vector_store():
    """Create a fresh vector store, replacing any existing one"""
    reset_vector_store()
    return get_qdrant_store()
