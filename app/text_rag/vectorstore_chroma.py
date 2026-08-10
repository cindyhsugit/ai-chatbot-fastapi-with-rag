import chromadb

# Initialize the client and collection at the module level
client = chromadb.PersistentClient(path="chroma_data")
collection = client.get_or_create_collection(name="documents")


def add_documents(embeddings: list[list[float]], documents: list[str]):
    """
    Handles ID generation and batch insertion into ChromaDB.
    """
    # Auto-generate IDs based on the length of documents currently being added
    ids = [str(i) for i in range(len(documents))]

    collection.add(ids=ids, embeddings=embeddings, documents=documents)

    # Debug print as requested
    print(f"ChromaDB Collection Count: {collection.count()}")


def search(query_embedding: list[float], k: int = 20) -> list[str]:
    """
    Returns the top k matching chunk texts for a given query embedding.
    """
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    # Return the documents list from the first result set
    return results["documents"][0] if results["documents"] else []
