# Technical Document QA Assistant using RAG

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) based question-answering system for technical documents.

Users can ask questions about technical research papers, and the system retrieves relevant document chunks and generates answers grounded in the retrieved information.

## Architecture

PDF Documents
↓
Text Extraction
↓
Document Chunking
↓
Sentence Transformer Embeddings
↓
ChromaDB Vector Database
↓
Semantic Retrieval
↓
Qwen2.5-1.5B-Instruct
↓
Answer Generation
↓
Source and Page Citations

## Technologies

- Python
- PyPDF
- Sentence Transformers
- ChromaDB
- Qwen2.5-1.5B-Instruct
- Hugging Face Transformers
- Streamlit

## Features

- Technical PDF document ingestion
- Semantic document retrieval
- Retrieval-Augmented Generation
- Context-grounded answers
- Source document identification
- Page-level source references
