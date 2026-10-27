
import chromadb


def create_chroma_collection(collection_name: str,embeddings,chunks: list,
    document_id: str,path: str = "./chroma_db") -> chromadb.Collection:

    client = chromadb.PersistentClient(path=path)

    # Reuse the collection instead of creating a new name each time
    collection = client.get_or_create_collection(name=collection_name)

    # Check whether this PDF is already stored
    existing = collection.get(where={"document_id": document_id},limit=1)

    if existing["ids"]:
        print("PDF already stored in ChromaDB.")
        return collection

    # Store chunks, embeddings, and the PDF hash
    collection.add(ids=[
            f"{document_id}_chunk_{i}"
            for i in range(len(chunks))
        ],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"document_id": document_id}
            for _ in chunks
        ]
    )

    print("PDF chunks added to ChromaDB.")
    return collection

