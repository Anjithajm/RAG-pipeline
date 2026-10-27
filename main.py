from load_data import load_pdf
from hash_id import get_hash_id
from chunking import chunk_text
from embedding import embed_chunks
from chromadb_store import create_chroma_collection
from top_chunk import top_score_chunks
from generation import generate_answer

if __name__ == "__main__":
    pdf_path = "./paper.pdf" 
    raw_pdf_text = load_pdf(pdf_path)
    print(f"PDF Loaded: Extracted {len(raw_pdf_text)} characters.")

    #hash id
    doc_id = get_hash_id(raw_pdf_text)
    print(f"Document ID (hash): {doc_id}")

    #chunking
    chunks = chunk_text(raw_pdf_text)
    print(f"Text Chunked: {len(chunks)} chunks created.")

    #tokenize and embed
    embeddings = embed_chunks(chunks)

    #store in ChromaDB
    collection = create_chroma_collection(collection_name="pdf_chunks",
                        embeddings=embeddings,chunks=chunks,document_id=doc_id)

    # embed the question
    question = input("Ask your question: ").strip()
    if question == "":
        question = "What is the main topic of the paper?"
    question_embedding = embed_chunks([question])[0]

    # find top chunk
    top_chunks = top_score_chunks(question_embedding,collection,top_k=3)

    # generate answer
    answer = generate_answer(question,top_chunks)
    print("Answer:", answer)

    


    





    

    
    
    