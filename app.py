import os
import io
import hashlib
import streamlit as st
import chromadb
from pypdf import PdfReader
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from groq import Groq


# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Technical Document QA Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# 2. TITLE
# ============================================================

st.title(
    "📚 Technical Document QA Assistant"
)

st.write(
    "Upload technical PDF documents and ask questions about them "
    "using Retrieval-Augmented Generation (RAG)."
)

st.caption(
    "Semantic retrieval + Cross-Encoder reranking + "
    "LLM-based grounded generation"
)


# ============================================================
# 3. LOAD EMBEDDING MODEL / RERANKER (these are fine to cache —
#    they don't depend on user-uploaded content, only on model name)
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_reranker():
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


try:
    embedding_model = load_embedding_model()
    reranker = load_reranker()
except Exception as e:
    st.error("Failed to load embedding/reranking models.")
    st.exception(e)
    st.stop()


# ============================================================
# 4. INITIALIZE GROQ CLIENT
# ============================================================

groq_client = None
groq_init_error = None

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    groq_client = Groq(api_key=groq_api_key)
except Exception as e:
    # Keep the real reason instead of silently collapsing it —
    # missing key, bad key, and import errors are different problems.
    groq_init_error = str(e)


# ============================================================
# 5. SESSION STATE
# ============================================================
# Everything tied to the CURRENT set of uploaded PDFs lives in
# session_state, scoped to this user's session — not on disk,
# and not shared across users. This is what makes "dynamic"
# actually dynamic instead of accumulating stale data.

if "collection" not in st.session_state:
    st.session_state.collection = None

if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

if "processed_files_hash" not in st.session_state:
    st.session_state.processed_files_hash = None

if "chroma_client" not in st.session_state:
    # Ephemeral (in-memory) client: no disk persistence, so there's
    # nothing to go stale between sessions and nothing to clean up.
    st.session_state.chroma_client = chromadb.EphemeralClient()


# ============================================================
# 6. FILE UPLOAD UI
# ============================================================

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

process_clicked = st.button(
    "📥 Process Documents",
    disabled=not uploaded_files
)


# ============================================================
# 7. HELPERS
# ============================================================

def hash_uploaded_files(files):
    """
    Fingerprint the exact set of uploaded files (name + bytes).
    Used to detect whether the user re-clicked 'Process' on the
    same files (skip reprocessing) or actually changed them
    (must reprocess).
    """
    hasher = hashlib.sha256()
    for f in files:
        f.seek(0)
        hasher.update(f.name.encode("utf-8"))
        hasher.update(f.read())
        f.seek(0)
    return hasher.hexdigest()


def extract_pdf_pages(uploaded_file):
    """Extract text per page from an in-memory uploaded PDF."""
    pages = []
    try:
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(
                    {
                        "text": text.strip(),
                        "source": uploaded_file.name,
                        "page": page_number + 1
                    }
                )
    except Exception as e:
        st.warning(f"Could not process {uploaded_file.name}: {e}")
    return pages


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = []
    for document in documents:
        split_texts = text_splitter.split_text(document["text"])
        for text in split_texts:
            chunks.append(
                {
                    "text": text,
                    "source": document["source"],
                    "page": document["page"]
                }
            )
    return chunks


def build_fresh_collection(chunks, files_hash):
    """
    Always start from a clean collection for the current upload
    batch. Using the files_hash in the collection name (rather than
    a fixed name) means re-processing never mixes old and new
    chunks, and reusing chunk_i-style positional IDs can't silently
    upsert over unrelated content.
    """
    client = st.session_state.chroma_client

    collection_name = f"session_docs_{files_hash[:16]}"

    # Clear out any previous collection from an earlier upload in
    # this session so old PDFs' chunks don't linger.
    for existing in client.list_collections():
        if existing.name != collection_name:
            client.delete_collection(existing.name)

    collection = client.get_or_create_collection(name=collection_name)

    if collection.count() == 0 and chunks:
        chunk_texts = [c["text"] for c in chunks]

        embeddings = embedding_model.encode(
            chunk_texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Content-derived IDs instead of positional chunk_i IDs —
        # stable and collision-resistant across reprocessing.
        ids = [
            hashlib.sha256(
                f"{c['source']}|{c['page']}|{i}|{c['text']}".encode("utf-8")
            ).hexdigest()
            for i, c in enumerate(chunks)
        ]

        metadatas = [
            {"source": c["source"], "page": c["page"]}
            for c in chunks
        ]

        collection.add(
            ids=ids,
            documents=chunk_texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

    return collection


def retrieve_documents(collection, query, top_k=10):
    query_embedding = embedding_model.encode(
        query, convert_to_numpy=True
    )
    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count())
    )


def rerank_documents(query, results, top_k=3):
    if not results:
        return []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return []

    pairs = [[query, document] for document in documents]
    scores = reranker.predict(pairs)

    ranked_results = []
    for document, metadata, score in zip(documents, metadatas, scores):
        ranked_results.append(
            {
                "text": document,
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page", "Unknown"),
                "score": float(score)
            }
        )

    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    return ranked_results[:top_k]


def build_context(reranked_documents):
    context_parts = []
    for i, document in enumerate(reranked_documents, start=1):
        context_parts.append(
            f"""[Context {i}]
Source: {document["source"]}
Page: {document["page"]}
Content:
{document["text"]}
"""
        )
    return "\n\n".join(context_parts)


def generate_answer(question, context):
    if groq_client is None:
        detail = f" ({groq_init_error})" if groq_init_error else ""
        return (
            "Groq client is not available"
            f"{detail}. Check GROQ_API_KEY in Streamlit Secrets."
        )

    system_prompt = """
You are a technical document question-answering assistant.

Rules:
1. Answer using ONLY the provided context.
2. Do not use outside knowledge or invent facts.
3. If the answer is not available in the context, say exactly:
   "I could not find the answer in the provided documents."
4. Give a clear and concise answer.
5. When possible, mention the relevant source and page.
"""

    user_prompt = f"""
Context:

{context}

Question:

{question}

Answer the question using only the context above.
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {e}"


# ============================================================
# 8. PROCESS UPLOADED FILES
# ============================================================

if process_clicked and uploaded_files:

    current_hash = hash_uploaded_files(uploaded_files)

    if current_hash == st.session_state.processed_files_hash:
        st.info("These files are already processed — no changes detected.")
    else:
        with st.spinner("Extracting text from PDFs..."):
            all_documents = []
            for uf in uploaded_files:
                all_documents.extend(extract_pdf_pages(uf))

        if not all_documents:
            st.error("No readable text found in the uploaded PDF(s).")
        else:
            with st.spinner("Chunking text..."):
                chunks = create_chunks(all_documents)

            with st.spinner("Embedding and indexing (this may take a moment)..."):
                collection = build_fresh_collection(chunks, current_hash)

            st.session_state.collection = collection
            st.session_state.chunks_count = len(chunks)
            st.session_state.processed_files_hash = current_hash

            st.success(
                f"Processed {len(uploaded_files)} file(s) into "
                f"{len(chunks)} chunks. Ready for questions."
            )


# ============================================================
# 9. QUESTION UI — only enabled once documents are processed
# ============================================================

st.divider()

if st.session_state.collection is None:
    st.info("Upload PDF(s) and click 'Process Documents' to begin.")
else:
    question = st.text_input(
        "Ask your question",
        placeholder="Example: What are the main components of a RAG system?"
    )

    if st.button("🔍 Ask Question"):

        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        try:
            with st.spinner("Retrieving and reranking documents..."):
                retrieved_results = retrieve_documents(
                    st.session_state.collection, question, top_k=10
                )
                reranked_documents = rerank_documents(
                    question, retrieved_results, top_k=3
                )
                context = build_context(reranked_documents)

            with st.spinner("Generating grounded answer..."):
                answer = generate_answer(question, context)

            st.subheader("Answer")
            st.write(answer)

            with st.expander("View Retrieved Context"):
                for i, document in enumerate(reranked_documents, start=1):
                    st.markdown(f"### Retrieved Passage {i}")
                    st.write(document["text"])
                    st.caption(
                        f"📄 {document['source']} "
                        f"| Page {document['page']} "
                        f"| Reranker Score: {document['score']:.4f}"
                    )
                    st.divider()

            st.subheader("Sources")
            shown_sources = set()
            for document in reranked_documents:
                citation = (document["source"], document["page"])
                if citation not in shown_sources:
                    st.write(f"📄 {document['source']} | Page {document['page']}")
                    shown_sources.add(citation)

        except Exception as e:
            st.error("An error occurred while processing your question.")
            st.exception(e)
