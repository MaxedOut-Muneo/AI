import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

LEGAL_COLLECTION_NAME = "interior_legal_docs"

embedding_function = embedding_functions.DefaultEmbeddingFunction()


def get_legal_collection():
    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_TENANT")
    database = os.getenv("CHROMA_DATABASE")

    if not api_key or not tenant or not database:
        raise ValueError("CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE가 .env에 필요합니다.")

    client = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database
    )

    return client.get_or_create_collection(
        name=LEGAL_COLLECTION_NAME,
        embedding_function=embedding_function
    )