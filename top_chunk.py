def top_score_chunks(query_embedding: np.ndarray, collection, top_k: int = 4) -> list[str]:
    # Pass the query embedding directly into ChromaDB's native search engine
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["documents", "distances"]
    )

    retrieved_chunks = results["documents"][0]
    distances = results["distances"][0]
    # print(retrieved_chunks)
    # for i, (chunk, distance) in enumerate(
    #     zip(retrieved_chunks, distances), start=1
    # ):
    #     print(f"\nChunk {i} | Distance: {distance:.4f}")
    #     print(chunk)


    return retrieved_chunks