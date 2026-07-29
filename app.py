import os
import re
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
    "Ask questions about technical research documents "
    "using Retrieval-Augmented Generation (RAG)."
)

st.caption(
    "Semantic retrieval + Cross-Encoder reranking + "
    "LLM-based grounded generation"
)


# ============================================================
# 3. PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PDF_FOLDER = os.path.join(
    BASE_DIR,
    "documents"
)

CHROMA_FOLDER = os.path.join(
    BASE_DIR,
    "chroma_db"
)


# ============================================================
# 4. CHECK DOCUMENT FOLDER
# ============================================================

if not os.path.exists(PDF_FOLDER):

    st.error(
        "The 'documents' folder was not found."
    )

    st.info(
        "Create a folder named 'documents' "
        "and place your PDF files inside it."
    )

    st.stop()


# ============================================================
# 5. LOAD PDF DOCUMENTS
# ============================================================

@st.cache_data
def load_documents():

    all_documents = []

    pdf_files = sorted(
        [
            file
            for file in os.listdir(PDF_FOLDER)
            if file.lower().endswith(".pdf")
        ]
    )

    if not pdf_files:

        return []

    for filename in pdf_files:

        file_path = os.path.join(
            PDF_FOLDER,
            filename
        )

        try:

            reader = PdfReader(
                file_path
            )

            for page_number, page in enumerate(
                reader.pages
            ):

                text = page.extract_text()

                if text and text.strip():

                    all_documents.append(
                        {
                            "text": text.strip(),
                            "source": filename,
                            "page": page_number + 1
                        }
                    )

        except Exception as e:

            st.warning(
                f"Could not process {filename}: {e}"
            )

    return all_documents


# ============================================================
# LOAD DOCUMENTS
# ============================================================

all_documents = load_documents()


if not all_documents:

    st.error(
        "No readable PDF documents were found."
    )

    st.stop()


st.success(
    f"Loaded {len(all_documents)} PDF pages."
)


# ============================================================
# 6. CREATE TEXT CHUNKS
# ============================================================

@st.cache_data
def create_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = []

    for document in documents:

        split_texts = text_splitter.split_text(
            document["text"]
        )

        for text in split_texts:

            chunks.append(
                {
                    "text": text,
                    "source": document["source"],
                    "page": document["page"]
                }
            )

    return chunks


chunks = create_chunks(
    all_documents
)


st.info(
    f"Created {len(chunks)} text chunks."
)


# ============================================================
# 7. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


try:

    embedding_model = (
        load_embedding_model()
    )

except Exception as e:

    st.error(
        "Failed to load embedding model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 8. LOAD RERANKER
# ============================================================

@st.cache_resource
def load_reranker():

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


try:

    reranker = load_reranker()

except Exception as e:

    st.error(
        "Failed to load reranking model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 9. CREATE / LOAD CHROMADB
# ============================================================

@st.cache_resource
def create_collection(chunks):

    client = chromadb.PersistentClient(
        path=CHROMA_FOLDER
    )

    collection = (
        client.get_or_create_collection(
            name="technical_documents"
        )
    )

    if collection.count() == 0:

        chunk_texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = (
            embedding_model.encode(
                chunk_texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        )

        ids = [
            f"chunk_{i}"
            for i in range(
                len(chunks)
            )
        ]

        metadatas = [
            {
                "source": chunk["source"],
                "page": chunk["page"]
            }
            for chunk in chunks
        ]

        collection.add(
            ids=ids,
            documents=chunk_texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

    return collection


try:

    collection = create_collection(
        chunks
    )

except Exception as e:

    st.error(
        "Failed to create or load ChromaDB."
    )

    st.exception(e)

    st.stop()


st.success(
    f"Vector database ready with "
    f"{collection.count()} chunks."
)


# ============================================================
# 10. INITIALIZE GROQ CLIENT
# ============================================================

try:

    groq_api_key = st.secrets[
        "GROQ_API_KEY"
    ]

    groq_client = Groq(
        api_key=groq_api_key
    )

except Exception:

    groq_client = None


# ============================================================
# 11. VECTOR RETRIEVAL
# ============================================================

def retrieve_documents(
    query,
    top_k=10
):

    query_embedding = (
        embedding_model.encode(
            query,
            convert_to_numpy=True
        )
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=min(
            top_k,
            collection.count()
        )
    )

    return results


# ============================================================
# 12. RERANK RETRIEVED DOCUMENTS
# ============================================================

def rerank_documents(
    query,
    results,
    top_k=3
):

    if not results:

        return []

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    if not documents:

        return []


    # Create query-document pairs

    pairs = [

        [
            query,
            document
        ]

        for document in documents

    ]


    # Get Cross-Encoder scores

    scores = reranker.predict(
        pairs
    )


    # Combine documents with scores

    ranked_results = []

    for document, metadata, score in zip(
        documents,
        metadatas,
        scores
    ):

        ranked_results.append(
            {
                "text": document,
                "source": metadata.get(
                    "source",
                    "Unknown"
                ),
                "page": metadata.get(
                    "page",
                    "Unknown"
                ),
                "score": float(score)
            }
        )


    # Sort by relevance score

    ranked_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    return ranked_results[
        :top_k
    ]


# ============================================================
# 13. BUILD CONTEXT
# ============================================================

def build_context(
    reranked_documents
):

    context_parts = []

    for i, document in enumerate(
        reranked_documents,
        start=1
    ):

        context_parts.append(

            f"""
[Context {i}]

Source:
{document["source"]}

Page:
{document["page"]}

Content:
{document["text"]}
"""

        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# 14. GENERATE ANSWER USING LLM API
# ============================================================

def generate_answer(
    question,
    context
):

    if groq_client is None:

        return (
            "Groq API key is not configured. "
            "Please add GROQ_API_KEY to Streamlit Secrets."
        )


    system_prompt = """

You are a technical document question-answering assistant.

Your task is to answer the user's question using ONLY
the provided context.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer is not available in the context,
   say exactly:

   "I could not find the answer in the provided documents."

4. Give a clear and concise answer.
5. Use the retrieved context as the source of truth.
6. When possible, mention the relevant source and page.

"""


    user_prompt = f"""

Context:

{context}


Question:

{question}


Answer the question using only the context above.

"""


    try:

        response = (
            groq_client.chat.completions.create(

                model=(
                    "llama-3.1-8b-instant"
                ),

                messages=[

                    {
                        "role":
                        "system",

                        "content":
                        system_prompt
                    },

                    {
                        "role":
                        "user",

                        "content":
                        user_prompt
                    }

                ],

                temperature=0.1,

                max_tokens=300

            )
        )


        answer = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )


        return answer


    except Exception as e:

        return (
            f"Error generating answer: {e}"
        )


# ============================================================
# 15. STREAMLIT USER INTERFACE
# ============================================================

st.divider()


question = st.text_input(

    "Ask your question",

    placeholder=(
        "Example: "
        "What are the main components of a RAG system?"
    )

)


if st.button(
    "🔍 Ask Question"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    try:

        with st.spinner(
            "Retrieving and reranking documents..."
        ):

            # ----------------------------------------------
            # STEP 1: VECTOR RETRIEVAL
            # ----------------------------------------------

            retrieved_results = (
                retrieve_documents(
                    question,
                    top_k=10
                )
            )


            # ----------------------------------------------
            # STEP 2: RERANKING
            # ----------------------------------------------

            reranked_documents = (
                rerank_documents(
                    question,
                    retrieved_results,
                    top_k=3
                )
            )


            # ----------------------------------------------
            # STEP 3: BUILD CONTEXT
            # ----------------------------------------------

            context = build_context(
                reranked_documents
            )


        # ====================================================
        # GENERATE ANSWER
        # ====================================================

        with st.spinner(
            "Generating grounded answer..."
        ):

            answer = generate_answer(
                question,
                context
            )


        # ====================================================
        # DISPLAY ANSWER
        # ====================================================

        st.subheader(
            "Answer"
        )

        st.write(
            answer
        )


        # ====================================================
        # DISPLAY RETRIEVED CONTEXT
        # ====================================================

        with st.expander(
            "View Retrieved Context"
        ):

            for i, document in enumerate(
                reranked_documents,
                start=1
            ):

                st.markdown(
                    f"### Retrieved Passage {i}"
                )

                st.write(
                    document["text"]
                )

                st.caption(

                    f"📄 {document['source']} "
                    f"| Page {document['page']} "
                    f"| Reranker Score: "
                    f"{document['score']:.4f}"

                )

                st.divider()


        # ====================================================
        # DISPLAY SOURCES
        # ====================================================

        st.subheader(
            "Sources"
        )


        shown_sources = set()


        for document in (
            reranked_documents
        ):

            source = document[
                "source"
            ]

            page = document[
                "page"
            ]


            citation = (
                source,
                page
            )


            if citation not in shown_sources:

                st.write(

                    f"📄 {source} "
                    f"| Page {page}"

                )

                shown_sources.add(
                    citation
                )


    except Exception as e:

        st.error(
            "An error occurred while "
            "processing your question."
        )

        st.exception(
            e
        )
