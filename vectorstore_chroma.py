import chromadb

# This creates a small local database saved to disk in a folder
# called "chroma_data" - so your data survives even after you
# restart your app (unlike FAISS, which lives only in memory).
client = chromadb.PersistentClient(path="chroma_data")

# A "collection" is like one drawer in the filing cabinet.
# get_or_create means: use it if it exists, make it if it doesn't.
collection = client.get_or_create_collection(name="documents")


def add_documents(ids, embeddings, documents):
    """
    ids: a list of unique string IDs, one per chunk, e.g. ["0", "1", "2"]
    embeddings: your HuggingFace embeddings (same as before)
    documents: the actual text of each chunk
    """
    collection.add(ids=ids, embeddings=embeddings, documents=documents)
    print("Total documents in collection:", collection.count())

def search(query_embedding, k=20):
    """
    query_embedding: the embedded user question
    k: how many results to bring back (same idea as FAISS's k)
    Returns: a list of the top k matching chunk texts
    """
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    # results["documents"][0] is the list of matching texts
    
    return results["documents"][0]
