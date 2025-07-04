from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
import os

def get_qdrant_store(collection_name="pdf_docs"):
    client = QdrantClient(url=os.getenv("QDRANT_URL"))
    
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return Qdrant(client=client, collection_name=collection_name, embeddings=embeddings)
