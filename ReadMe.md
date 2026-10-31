# PDF Q&A with Local RAG

A retrieval-augmented generation (RAG) pipeline that answers questions about a PDF using only what the PDF says. Everything runs locally: PyMuPDF extracts the text, Sentence-Transformers embeds it, ChromaDB stores it, and Llama 3.2 (3B) running in Ollama writes the answer. Each answer cites the excerpts it used, with page numbers.

The default document is `paper.pdf`. Any text-based PDF works.

## How it works


1. **Load**: PyMuPDF extracts the text of each page and keeps its page number.
2. **Fingerprint**: a SHA-256 hash of the full extracted text becomes the document ID. It is used to avoid storing the same PDF twice and to keep searches inside the current document.
3. **Chunk**: `RecursiveCharacterTextSplitter` splits each page into chunks of about 500 characters with a 50-character overlap, breaking at paragraphs first, then lines, then words. Every chunk keeps its page number.
4. **Embed**: each chunk is turned into a 384-dimensional vector with `all-MiniLM-L6-v2`.
5. **Store**: chunks, vectors and metadata (`document_id`, `page_num`) go into the persistent ChromaDB collection `pdf_chunks`. If that document ID is already in the collection, nothing new is added.
6. **Retrieve**: the question is embedded with the same model, and the 3 closest chunks from the current document are returned.
7. **Generate**: the numbered excerpts and the question are sent to `llama3.2:3b` with a strict system prompt: use only the context, cite excerpts as `[1]`, `[2]`, point out conflicting excerpts, and reply *"I don't know based on the provided context."* when the answer isn't there. Temperature is 0 and the context window is 8192 tokens.
8. **Cite**: the answer is returned with a list of which page each excerpt came from. If the model says it doesn't know, no sources are shown.

## Project structure

```
.
├── main.py              # Runs the full pipeline end to end
├── load_data.py         # PDF -> list of {page_num, page_content}
├── hash_id.py           # SHA-256 document ID from the extracted text
├── chunking.py          # Page-aware text chunking
├── embedding.py         # Sentence-Transformers embedding model
├── chromadb_store.py    # Persistent ChromaDB collection with de-duplication
├── top_chunk.py         # Similarity search scoped to one document
├── generation.py        # Prompts Llama 3.2 via Ollama and adds sources
├── paper.pdf            # The document to query
└── chroma_db/           # Vector store, created on the first run
```

## Requirements

- Python 3.9 or newer
- [Ollama](https://ollama.com) installed and running
- About 2 GB of disk space for `llama3.2:3b`, plus about 90 MB for the embedding model

## Installation

```bash
git clone <your-repo-url>
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install pymupdf langchain-text-splitters sentence-transformers chromadb ollama

ollama pull llama3.2:3b
```

The embedding model is downloaded from Hugging Face automatically the first time you run the script.

## Usage

Start Ollama (the desktop app, or `ollama serve`), then run:

```bash
python main.py
```

The script loads and indexes the PDF, then asks for a question. Press Enter without typing anything to use the default question, *"What is the main topic of the paper?"*

Output looks like this:

```
PDF Loaded: 14 pages.
Document ID (hash): <64-character hash>
Text Chunked: <N> chunks created.
PDF chunks added to ChromaDB.
Hi there! Ask your question :) <your question>
Answer: <answer citing [1], [2], ...>

Sources:
[1] page <n>
[2] page <n>
[3] page <n>
```

On later runs with the same PDF you'll see `PDF already stored in ChromaDB.` instead of the "added" line.

### Example questions for the Hybrid-z paper

- What is the main topic of the paper?
- What loss function is used to train Hybrid-z, and why?
- How much does Hybrid-z reduce photo-z scatter compared with ANNz2?
- Why was the GAMA training set smoothed before the final predictions?
- What image cutout size was used for the galaxies?

### Using a different PDF

Change `pdf_path` in `main.py`. Several PDFs can live in the same collection: each is stored once under its own document ID, and a query only searches the PDF currently loaded.

## Configuration

| Setting | Where | Default |
|---|---|---|
| PDF to query | `main.py`, `pdf_path` | `./paper.pdf` |
| Chunk size / overlap | `chunking.py` | 500 / 50 characters |
| Embedding model | `embedding.py` | `all-MiniLM-L6-v2` |
| Vector store location | `chromadb_store.py`, `path` | `./chroma_db` |
| Collection name | `main.py`, `collection_name` | `pdf_chunks` |
| Chunks retrieved per question | `main.py`, `top_k` | 3 |
| Language model | `generation.py`, `model` | `llama3.2:3b` |
| Temperature / context window | `generation.py`, `options` | 0.0 / 8192 |

**After changing the chunking settings or the embedding model, reset the vector store** (see below) or use a new collection name. The duplicate check only looks at the document ID, so a PDF that is already stored keeps its old chunks and vectors, and question vectors from a different model won't match them.

## Resetting the vector store

Delete the database folder; it is rebuilt on the next run.

```bash
rm -rf chroma_db          # Windows: rmdir /s /q chroma_db
```

## Known limitations

- **Embeddings are recomputed on every run**, even when the PDF is already stored, because the duplicate check in `chromadb_store.py` happens after `embed_chunks`. Checking for the document ID before embedding would make repeat runs faster.
- **The sources list shows every retrieved excerpt**, not only the ones the model actually cited.
- **The "I don't know" check is a substring match**, so any answer containing that phrase also loses its sources.
- **Text only**: `page.get_text()` flattens tables and equations and ignores figures. Scanned PDFs need OCR first.
- **One question per run**: the script exits after answering.
- **Small model**: a 3B model can still misread or mis-cite an excerpt, so check the cited pages for anything important.

