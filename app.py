import os
import re
import streamlit as st
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


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

st.title("📚 Technical Document QA Assistant")

st.write(
    "Ask questions about technical research documents "
    "using Retrieval-Augmented Generation (RAG)."
)

st.caption(
    "The system retrieves the most relevant passages "
    "from your uploaded technical documents."
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


# Load documents

all_documents = load_documents()


# ============================================================
# 6. CHECK DOCUMENTS
# ============================================================

if not all_documents:

    st.error(
        "No readable PDF documents were found."
    )

    st.stop()


st.success(
    f"Loaded {len(all_documents)} PDF pages."
)


# ============================================================
# 7. CREATE TEXT CHUNKS
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
# 8. LOAD SENTENCE TRANSFORMER EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


try:

    embedding_model = load_embedding_model()

except Exception as e:

    st.error(
        "Failed to load embedding model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 9. CREATE / LOAD CHROMADB
# ============================================================

@st.cache_resource
def create_collection(chunks):

    # Persistent ChromaDB

    client = chromadb.PersistentClient(
        path=CHROMA_FOLDER
    )

    collection = client.get_or_create_collection(
        name="technical_documents"
    )

    # Add data only if collection is empty

    if collection.count() == 0:

        chunk_texts = [
            chunk["text"]
            for chunk in chunks
        ]

        # Generate embeddings

        embeddings = embedding_model.encode(
            chunk_texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # IDs

        ids = [
            f"chunk_{i}"
            for i in range(len(chunks))
        ]

        # Metadata

        metadatas = [
            {
                "source": chunk["source"],
                "page": chunk["page"]
            }
            for chunk in chunks
        ]

        # Store in ChromaDB

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
# 10. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(
    query,
    top_k=3
):

    # Convert query to embedding

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True
    )

    # Search ChromaDB

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
# 11. SIMPLE EXTRACTIVE ANSWER GENERATION
# ============================================================

def generate_answer(
    question,
    retrieved_documents
):

    if (
        not retrieved_documents
        or "documents" not in retrieved_documents
    ):

        return (
            "I could not find the answer "
            "in the provided documents."
        )


    documents = retrieved_documents[
        "documents"
    ][0]


    if not documents:

        return (
            "I could not find the answer "
            "in the provided documents."
        )


    # --------------------------------------------------------
    # Extract important keywords from question
    # --------------------------------------------------------

    question_words = set(

        re.findall(

            r"\b[a-zA-Z]{3,}\b",

            question.lower()

        )

    )


    scored_sentences = []


    # --------------------------------------------------------
    # Score sentences based on keyword overlap
    # --------------------------------------------------------

    for document in documents:

        sentences = re.split(

            r"(?<=[.!?])\s+",

            document

        )


        for sentence in sentences:

            sentence = sentence.strip()


            if not sentence:

                continue


            sentence_words = set(

                re.findall(

                    r"\b[a-zA-Z]{3,}\b",

                    sentence.lower()

                )

            )


            overlap = (

                question_words

                & sentence_words

            )


            score = len(overlap)


            if score > 0:

                scored_sentences.append(

                    (
                        score,

                        sentence

                    )

                )


    # --------------------------------------------------------
    # If no matching sentence
    # --------------------------------------------------------

    if not scored_sentences:

        return (

            "I could not find a direct answer "
            "in the retrieved document passages."
        )


    # --------------------------------------------------------
    # Sort by relevance
    # --------------------------------------------------------

    scored_sentences.sort(

        key=lambda x: x[0],

        reverse=True

    )


    # --------------------------------------------------------
    # Select top sentences
    # --------------------------------------------------------

    selected_sentences = []

    seen = set()


    for score, sentence in scored_sentences:

        normalized = sentence.lower()


        if normalized not in seen:

            selected_sentences.append(
                sentence
            )

            seen.add(
                normalized
            )


        if len(
            selected_sentences
        ) >= 4:

            break


    answer = " ".join(
        selected_sentences
    )


    return answer


# ============================================================
# 12. STREAMLIT USER INTERFACE
# ============================================================

st.divider()


question = st.text_input(

    "Ask your question",

    placeholder=(
        "Example: "
        "What is Retrieval-Augmented Generation?"
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
            "Searching technical documents..."
        ):

            # Retrieve relevant chunks

            results = retrieve_documents(

                question,

                top_k=3

            )


            # Generate answer from retrieved context

            answer = generate_answer(

                question,

                results

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

            if results.get(
                "documents"
            ):

                for i, document in enumerate(

                    results["documents"][0],

                    start=1

                ):

                    st.markdown(

                        f"**Retrieved Passage {i}**"

                    )

                    st.write(
                        document
                    )

                    st.divider()


        # ====================================================
        # DISPLAY SOURCES
        # ====================================================

        st.subheader(
            "Sources"
        )


        shown_sources = set()


        if (

            results.get(
                "metadatas"
            )

            and results["metadatas"][0]

        ):

            for metadata in (

                results["metadatas"][0]

            ):

                source = metadata.get(
                    "source",
                    "Unknown"
                )

                page = metadata.get(
                    "page",
                    "Unknown"
                )


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
