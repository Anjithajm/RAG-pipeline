from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(loaded_pdf: str) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(loaded_pdf)

