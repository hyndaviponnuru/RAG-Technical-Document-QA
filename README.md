

# RAG Technical Document QA Assistant

A Retrieval-Augmented Generation system for question answering over technical research documents.

## Architecture

PDF Documents
→ Text Extraction
→ Chunking
→ Sentence Transformer Embeddings
→ ChromaDB Vector Database
→ Semantic Retrieval
→ Qwen2.5-1.5B-Instruct
→ Answer Generation
→ Source Citations

## Technologies

- Python
- Streamlit
- LangChain
- PyPDF
- Sentence Transformers
- ChromaDB
- Qwen2.5-1.5B-Instruct
- Hugging Face Transformers

## Features

- Upload and process technical PDF documents
- Semantic search using vector embeddings
- Retrieval-Augmented Generation
- Context-grounded answers
- Source document and page references
