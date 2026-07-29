import os
import hashlib

import streamlit as st
import chromadb
import fitz

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
    "Semantic Retrieval + Cross-Encoder Reranking + "
    "LLM-Based Grounded Generation"
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


# Create documents folder if it doesn't exist

os.makedirs(
    PDF_FOLDER,
    exist_ok=True
)


# ============================================================
# 4. TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)


# ============================================================
# 5. LOAD EMBEDDING MODEL
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
# 6. LOAD CROSS-ENCODER RERANKER
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
# 7. CREATE / LOAD CHROMADB
# ============================================================

@st.cache_resource
def load_chroma():

    client = chromadb.PersistentClient(
        path=CHROMA_FOLDER
    )

    collection = (
        client.get_or_create_collection(
            name="technical_documents"
        )
    )

    return client, collection


try:

    chroma_client, collection = (
        load_chroma()
    )

except Exception as e:

    st.error(
        "Failed to create or load ChromaDB."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 8. PDF PROCESSING FUNCTION
# ============================================================

def process_pdf(
    pdf_path,
    source_name
):

    documents = []

    try:

        reader = PdfReader(
            pdf_path
        )

        for page_number, page in enumerate(
            reader.pages
        ):

            text = page.extract_text()

            if text and text.strip():

                documents.append(
                    {
                        "text": text.strip(),
                        "source": source_name,
                        "page": page_number + 1
                    }
                )

    except Exception as e:

        st.error(
            f"Error reading {source_name}: {e}"
        )

        return []


    # ========================================================
    # CREATE CHUNKS
    # ========================================================

    chunks = []

    for document in documents:

        split_texts = (
            text_splitter.split_text(
                document["text"]
            )
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


# ============================================================
# 9. ADD CHUNKS TO CHROMADB
# ============================================================

def add_chunks_to_database(
    chunks
):

    if not chunks:

        return 0


    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]


    # Generate embeddings

    embeddings = (
        embedding_model.encode(
            chunk_texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
    )


    ids = []

    metadatas = []


    for i, chunk in enumerate(
        chunks
    ):

        # Create unique ID using content

        content_hash = hashlib.md5(
            chunk["text"].encode(
                "utf-8"
            )
        ).hexdigest()


        chunk_id = (
            f"{chunk['source']}_"
            f"{chunk['page']}_"
            f"{content_hash}_"
            f"{i}"
        )


        ids.append(
            chunk_id
        )


        metadatas.append(
            {
                "source":
                chunk["source"],

                "page":
                chunk["page"]
            }
        )


    # Add to ChromaDB

    collection.add(

        ids=ids,

        documents=chunk_texts,

        embeddings=embeddings.tolist(),

        metadatas=metadatas

    )


    return len(
        chunks
    )


# ============================================================
# 10. LOAD EXISTING PDF DOCUMENTS
# ============================================================

def load_existing_documents():

    pdf_files = [

        file

        for file in os.listdir(
            PDF_FOLDER
        )

        if file.lower().endswith(
            ".pdf"
        )

    ]

    return pdf_files


# ============================================================
# 11. SIDEBAR
# ============================================================

st.sidebar.header(
    "📄 Document Management"
)


# ============================================================
# 12. PDF UPLOAD
# ============================================================

uploaded_file = (
    st.sidebar.file_uploader(

        "Upload a PDF document",

        type=["pdf"],

        help=(
            "Upload a technical research "
            "paper or PDF document."
        )

    )
)


# ============================================================
# 13. PROCESS UPLOADED PDF
# ============================================================
def process_pdf(pdf_path, source_name):

    documents = []

    try:

        # Open PDF using PyMuPDF
        pdf_document = fitz.open(pdf_path)

        for page_number in range(
            len(pdf_document)
        ):

            page = pdf_document.load_page(
                page_number
            )

            # Extract text
            text = page.get_text(
                "text"
            )

            if text and text.strip():

                documents.append(
                    {
                        "text": text.strip(),
                        "source": source_name,
                        "page": page_number + 1
                    }
                )

        pdf_document.close()

    except Exception as e:

        st.error(
            f"Error reading {source_name}: {e}"
        )

        return []


    # ========================================================
    # CREATE CHUNKS
    # ========================================================

    chunks = []

    for document in documents:

        split_texts = (
            text_splitter.split_text(
                document["text"]
            )
        )

        for text in split_texts:

            if text.strip():

                chunks.append(
                    {
                        "text": text.strip(),
                        "source": document["source"],
                        "page": document["page"]
                    }
                )

    return chunks

# ============================================================
# 14. DISPLAY DATABASE STATUS
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "📊 Knowledge Base"
)


st.sidebar.write(

    f"Total chunks: "
    f"**{collection.count()}**"

)


existing_files = (
    load_existing_documents()
)


st.sidebar.write(

    f"PDF files: "
    f"**{len(existing_files)}**"

)


# ============================================================
# 15. RETRIEVAL FUNCTION
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
# 16. RERANKING FUNCTION
# ============================================================

def rerank_documents(

    query,

    results,

    top_k=3

):

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


    pairs = [

        [

            query,

            document

        ]

        for document in documents

    ]


    # Cross-Encoder scores

    scores = reranker.predict(

        pairs

    )


    ranked_results = []


    for document, metadata, score in zip(

        documents,

        metadatas,

        scores

    ):

        ranked_results.append(

            {

                "text":
                document,

                "source":
                metadata.get(

                    "source",

                    "Unknown"

                ),

                "page":
                metadata.get(

                    "page",

                    "Unknown"

                ),

                "score":
                float(score)

            }

        )


    # Sort by relevance

    ranked_results.sort(

        key=lambda x:
        x["score"],

        reverse=True

    )


    return ranked_results[

        :top_k

    ]


# ============================================================
# 17. BUILD CONTEXT
# ============================================================

def build_context(

    documents

):

    context_parts = []


    for i, document in enumerate(

        documents,

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
# 18. LOAD GROQ API
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
# 19. GENERATE ANSWER
# ============================================================

def generate_answer(

    question,

    context

):

    if groq_client is None:

        return (

            "Groq API key is not configured. "
            "Please add GROQ_API_KEY to "
            "Streamlit Secrets."

        )


    system_prompt = """

You are a technical document question-answering assistant.

Answer the user's question ONLY using the provided context.

Rules:

1. Do not use outside knowledge.

2. Do not invent facts.

3. If the answer cannot be found in the provided
context, say:

"I could not find the answer in the provided documents."

4. Give a clear and concise answer.

5. Mention the relevant source and page when possible.

"""


    user_prompt = f"""

Context:

{context}


Question:

{question}


Answer the question using only the provided context.

"""


    try:

        response = (

            groq_client

            .chat

            .completions

            .create(

                model=
                "llama-3.1-8b-instant",

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
# 20. QUESTION INPUT
# ============================================================

st.divider()


question = st.text_input(

    "Ask your question",

    placeholder=(

        "Example: "
        "What is Retrieval-Augmented Generation?"

    )

)


# ============================================================
# 21. ASK QUESTION
# ============================================================

if st.button(

    "🔍 Ask Question"

):

    if not question.strip():

        st.warning(

            "Please enter a question."

        )

        st.stop()


    if collection.count() == 0:

        st.error(

            "The knowledge base is empty. "
            "Please upload a PDF first."

        )

        st.stop()


    try:

        # ====================================================
        # STEP 1: RETRIEVE
        # ====================================================

        with st.spinner(

            "Searching the knowledge base..."

        ):

            retrieved_results = (

                retrieve_documents(

                    question,

                    top_k=10

                )

            )


        # ====================================================
        # STEP 2: RERANK
        # ====================================================

        with st.spinner(

            "Reranking relevant passages..."

        ):

            reranked_documents = (

                rerank_documents(

                    question,

                    retrieved_results,

                    top_k=3

                )

            )


        if not reranked_documents:

            st.warning(

                "No relevant documents were found."

            )

            st.stop()


        # ====================================================
        # STEP 3: BUILD CONTEXT
        # ====================================================

        context = build_context(

            reranked_documents

        )


        # ====================================================
        # STEP 4: GENERATE ANSWER
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
        # RETRIEVED CONTEXT
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

                    f"📄 "
                    f"{document['source']} "
                    f"| Page "
                    f"{document['page']} "
                    f"| Reranker Score: "
                    f"{document['score']:.4f}"

                )


                st.divider()


        # ====================================================
        # SOURCES
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


            if citation not in (

                shown_sources

            ):


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
