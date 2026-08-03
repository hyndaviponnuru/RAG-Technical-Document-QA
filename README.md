# 📚 RAG Technical Document QA Assistant

A Retrieval-Augmented Generation (RAG) application that enables users to upload technical PDF documents and ask natural language questions. The system retrieves the most relevant document passages using semantic search and Cross-Encoder reranking, then generates grounded answers using the Groq Llama-3.1-8B model.

---

## 🚀 Features

- 📄 Upload one or multiple PDF documents
- 🔍 Automatic text extraction using PyPDF
- ✂️ Recursive text chunking
- 🧠 Semantic embeddings using Sentence Transformers
- 📚 ChromaDB vector database
- 🎯 Cross-Encoder reranking for improved retrieval
- 🤖 Grounded answer generation using Groq Llama-3.1-8B
- 📖 Source citations with page numbers
- ⚡ Fast Streamlit interface
- 🔄 Dynamic document processing with session state
- 🛡️ Duplicate upload detection using SHA-256 hashing

---

## 🏗️ Architecture

```
                PDF Documents
                      │
                      ▼
             Text Extraction (PyPDF)
                      │
                      ▼
        Recursive Text Chunking
       (Chunk Size=800, Overlap=100)
                      │
                      ▼
 SentenceTransformer Embeddings
     (all-MiniLM-L6-v2)
                      │
                      ▼
           ChromaDB Vector Store
                      │
                      ▼
            User Question Input
                      │
                      ▼
        Semantic Similarity Search
               (Top-10 Results)
                      │
                      ▼
        Cross-Encoder Reranking
      (Top-3 Most Relevant Chunks)
                      │
                      ▼
             Context Construction
                      │
                      ▼
      Groq Llama-3.1-8B-Instant
                      │
                      ▼
          Grounded Answer Generation
                      │
                      ▼
      Answer + Source Citations
```

---

## 🔄 Workflow

1. Upload one or more PDF documents.
2. Extract text from every page.
3. Split text into overlapping chunks.
4. Generate vector embeddings using Sentence Transformers.
5. Store embeddings in ChromaDB.
6. Convert the user's question into an embedding.
7. Retrieve the Top-10 most relevant chunks.
8. Rerank results using a Cross-Encoder.
9. Build contextual information from the Top-3 passages.
10. Generate an answer using Groq Llama-3.1-8B.
11. Display the answer with document and page citations.

---

## 🛠️ Tech Stack

### Frontend

- Streamlit

### NLP

- Sentence Transformers
- Cross Encoder
- LangChain Text Splitters

### Vector Database

- ChromaDB

### LLM

- Groq API
- Llama-3.1-8B-Instant

### PDF Processing

- PyPDF

### Language

- Python

---

## 📂 Project Structure

```
RAG-Technical-Document-QA/
│
├── app.py
├── evaluate_rag.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── .streamlit/
│   └── secrets.toml
│
└── sample_documents/
```

---

## 🤖 Models Used

### Embedding Model

```
sentence-transformers/all-MiniLM-L6-v2
```

### Reranker

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

### LLM

```
Llama-3.1-8B-Instant (Groq)
```

---

## 📊 Example Questions

- What is Retrieval-Augmented Generation?
- What are embeddings used for?
- Explain Sentence-BERT.
- What is LoRA?
- What are the advantages of RAG?

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/RAG-Technical-Document-QA.git

cd RAG-Technical-Document-QA

pip install -r requirements.txt

streamlit run app.py
```

---

## 🔑 Environment Variables

Create:

```
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY="YOUR_GROQ_API_KEY"
```
