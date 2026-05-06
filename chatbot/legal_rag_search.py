from chatbot.chroma_legal_client import get_legal_collection


def search_legal_docs(question: str, top_k: int = 4):
    collection = get_legal_collection()

    result = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = []

    if not result.get("documents") or not result["documents"][0]:
        return docs

    for i in range(len(result["documents"][0])):
        docs.append({
            "document": result["documents"][0][i],
            "metadata": result["metadatas"][0][i],
            "distance": result["distances"][0][i]
        })

    return docs