
#tokenize and embed the chunks
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks: list) -> list:
    return model.encode(chunks)



