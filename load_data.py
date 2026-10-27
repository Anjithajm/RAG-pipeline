import pymupdf

def load_pdf(pdf_path:str) -> str:
    pdf_document = pymupdf.open(pdf_path)
    full_text = ""
    for page in pdf_document:
        text = page.get_text()
        full_text += text + "\n"
    return full_text