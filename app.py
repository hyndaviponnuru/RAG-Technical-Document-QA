import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
import streamlit as st
import chromadb
import torch

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


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
        "Document folder not found."
    )

    st.info(
        "Create a 'documents' folder in your project "
        "and add PDF files inside it."
    )

    st.stop()


# ============================================================
# 5. LOAD PDF DOCUMENTS
# ============================================================

@st.cache_data
def load_documents():

    all_documents = []

    pdf_files = [

        file

        for file in os.listdir(
            PDF_FOLDER
        )

        if file.lower().endswith(
            ".pdf"
        )

    ]


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

                            "text":
                            text.strip(),

                            "source":
                            filename,

                            "page":
                            page_number + 1

                        }

                    )


        except Exception as e:

            st.warning(

                f"Could not process "
                f"{filename}: {e}"

            )


    return all_documents


# Load PDF documents

all_documents = load_documents()


# ============================================================
# CHECK DOCUMENTS
# ============================================================

if not all_documents:

    st.error(

        "No readable PDF documents were found "
        "inside the 'documents' folder."

    )

    st.stop()


st.success(

    f"Loaded {len(all_documents)} PDF pages."

)


# ============================================================
# 6. CHUNK DOCUMENTS
# ============================================================

@st.cache_data
def create_chunks(
    documents
):

    text_splitter = (

        RecursiveCharacterTextSplitter(

            chunk_size=800,

            chunk_overlap=100

        )

    )


    chunks = []


    for doc in documents:

        split_texts = (

            text_splitter.split_text(

                doc["text"]

            )

        )


        for text in split_texts:

            chunks.append(

                {

                    "text":
                    text,

                    "source":
                    doc["source"],

                    "page":
                    doc["page"]

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

    model = (

        SentenceTransformer(

            "sentence-transformers/"
            "all-MiniLM-L6-v2"

        )

    )


    return model


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
# 8. CREATE / LOAD CHROMADB
# ============================================================

@st.cache_resource
def create_collection(
    chunks
):

    # Create persistent ChromaDB client

    client = (

        chromadb.PersistentClient(

            path=CHROMA_FOLDER

        )

    )


    # Create or load collection

    collection = (

        client.get_or_create_collection(

            name="technical_documents"

        )

    )


    # Add documents only if empty

    if collection.count() == 0:

        chunk_texts = [

            chunk["text"]

            for chunk in chunks

        ]


        # Generate embeddings

        chunk_embeddings = (

            embedding_model.encode(

                chunk_texts,

                show_progress_bar=False,

                convert_to_numpy=True

            )

        )


        # Generate unique IDs

        chunk_ids = [

            f"chunk_{i}"

            for i in range(

                len(chunks)

            )

        ]


        # Metadata

        metadatas = [

            {

                "source":
                chunk["source"],

                "page":
                chunk["page"]

            }

            for chunk in chunks

        ]


        # Store documents and embeddings

        collection.add(

            ids=chunk_ids,

            documents=chunk_texts,

            embeddings=(
                chunk_embeddings.tolist()
            ),

            metadatas=metadatas

        )


    return collection


try:

    collection = (

        create_collection(

            chunks

        )

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
# 9. RETRIEVAL FUNCTION
# ============================================================

def retrieve_documents(

    query,

    top_k=3

):

    # Convert user question to embedding

    query_embedding = (

        embedding_model.encode(

            query,

            convert_to_numpy=True

        )

    )


    # Search ChromaDB

    results = (

        collection.query(

            query_embeddings=[

                query_embedding.tolist()

            ],

            n_results=min(

                top_k,

                collection.count()

            )

        )

    )


    return results


# ============================================================
# 10. LOAD QWEN LLM
# ============================================================

@st.cache_resource
def load_llm():

    model_name = (

        "Qwen/Qwen2.5-1.5B-Instruct"

    )


    # Use CPU on Streamlit deployment

    device = "cpu"


    st.info(

        "Loading language model. "
        "This may take some time on first startup."

    )


    # Load tokenizer

    tokenizer = (

        AutoTokenizer.from_pretrained(

            model_name

        )

    )


    # CPU uses float32

    model = (

        AutoModelForCausalLM.from_pretrained(

            model_name,

            dtype=torch.float32

        )

    )


    # Move model to CPU

    model = model.to(

        device

    )


    # Evaluation mode

    model.eval()


    return (

        tokenizer,

        model,

        device

    )


# ============================================================
# 11. BUILD CONTEXT
# ============================================================

def build_context(

    results

):

    context_parts = []


    if (

        not results

        or "documents"

        not in results

    ):

        return ""


    if not results["documents"]:

        return ""


    if not results["documents"][0]:

        return ""


    for text in results["documents"][0]:

        context_parts.append(

            text

        )


    return "\n\n".join(

        context_parts

    )


# ============================================================
# 12. RAG ANSWER FUNCTION
# ============================================================

def rag_answer(

    question,

    top_k=3

):

    # ========================================================
    # STEP 1: RETRIEVE DOCUMENTS
    # ========================================================

    results = (

        retrieve_documents(

            question,

            top_k=top_k

        )

    )


    # ========================================================
    # STEP 2: BUILD CONTEXT
    # ========================================================

    context = (

        build_context(

            results

        )

    )


    if not context:

        return (

            "I could not find the answer "
            "in the provided documents.",

            results

        )


    # Keep context manageable

    context = context[:4000]


    # ========================================================
    # STEP 3: LOAD LLM
    # ========================================================

    try:

        (

            tokenizer,

            model,

            device

        ) = load_llm()


    except Exception as e:

        raise RuntimeError(

            f"Failed to load Qwen model: {e}"

        )


    # ========================================================
    # STEP 4: CREATE PROMPT
    # ========================================================

    messages = [

        {

            "role":
            "system",

            "content":
            """
You are a technical document
question-answering assistant.

Answer the user's question using
ONLY the provided context.

Do not use outside knowledge.

If the answer is not available
in the context, say:

I could not find the answer
in the provided documents.

Give a clear and concise answer.
"""

        },

        {

            "role":
            "user",

            "content":
            f"""
Context:

{context}


Question:

{question}
"""

        }

    ]


    # ========================================================
    # STEP 5: CREATE CHAT PROMPT
    # ========================================================

    text = (

        tokenizer.apply_chat_template(

            messages,

            tokenize=False,

            add_generation_prompt=True

        )

    )


    # ========================================================
    # STEP 6: TOKENIZE
    # ========================================================

    inputs = (

        tokenizer(

            text,

            return_tensors="pt",

            truncation=True,

            max_length=1024

        )

    )


    # Move tensors to CPU

    inputs = {

        key:
        value.to(device)

        for key, value in inputs.items()

    }


    # ========================================================
    # STEP 7: GENERATE ANSWER
    # ========================================================

    try:

        with torch.no_grad():

            outputs = (

                model.generate(

                    **inputs,

                    max_new_tokens=100,

                    do_sample=False,

                    pad_token_id=(

                        tokenizer.eos_token_id

                    )

                )

            )


    except Exception as e:

        raise RuntimeError(

            f"Error during model generation: {e}"

        )


    # ========================================================
    # STEP 8: EXTRACT GENERATED TOKENS
    # ========================================================

    generated_tokens = (

        outputs[0][

            inputs["input_ids"].shape[1]:

        ]

    )


    # ========================================================
    # STEP 9: DECODE ANSWER
    # ========================================================

    answer = (

        tokenizer.decode(

            generated_tokens,

            skip_special_tokens=True

        )

        .strip()

    )


    if not answer:

        answer = (

            "I could not generate an answer "
            "from the provided documents."

        )


    return (

        answer,

        results

    )


# ============================================================
# 13. STREAMLIT USER INTERFACE
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

    if question.strip():

        try:

            with st.spinner(

                "Searching documents "
                "and generating answer..."

            ):

                answer, results = (

                    rag_answer(

                        question,

                        top_k=3

                    )

                )


            # =================================================
            # DISPLAY ANSWER
            # =================================================

            st.subheader(

                "Answer"

            )


            st.write(

                answer

            )


            # =================================================
            # DISPLAY SOURCES
            # =================================================

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

                    source = (

                        metadata.get(

                            "source",

                            "Unknown"

                        )

                    )


                    page = (

                        metadata.get(

                            "page",

                            "Unknown"

                        )

                    )


                    citation = (

                        source,

                        page

                    )


                    if (

                        citation

                        not in shown_sources

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

                "❌ An error occurred "
                "while processing your question."

            )


            st.exception(

                e

            )


    else:

        st.warning(

            "Please enter a question."

        )
