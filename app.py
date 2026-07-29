import os
import streamlit as st
import chromadb
import torch

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
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


# ============================================================
# 3. PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
        f"Document folder not found: {PDF_FOLDER}"
    )

    st.info(
        "Please create a 'documents' folder "
        "and add PDF files to it."
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
        for file in os.listdir(PDF_FOLDER)
        if file.lower().endswith(".pdf")
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

                    all_documents.append({

                        "text": text.strip(),

                        "source": filename,

                        "page": page_number + 1

                    })

        except Exception as e:

            st.warning(
                f"Could not process {filename}: {e}"
            )

    return all_documents


# Load documents

all_documents = load_documents()


# Check if documents exist

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
def create_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=800,

        chunk_overlap=100

    )

    chunks = []

    for doc in documents:

        split_texts = text_splitter.split_text(
            doc["text"]
        )

        for text in split_texts:

            chunks.append({

                "text": text,

                "source": doc["source"],

                "page": doc["page"]

            })

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

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


try:

    embedding_model = load_embedding_model()

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
    chunks,
    embedding_model
):

    # Create persistent ChromaDB client

    client = chromadb.PersistentClient(
        path=CHROMA_FOLDER
    )

    # Create or load collection

    collection = client.get_or_create_collection(

        name="technical_documents"

    )


    # Add documents only if collection is empty

    if collection.count() == 0:

        st.info(
            "Creating document embeddings. "
            "This may take some time on first startup..."
        )

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


        # Generate IDs

        chunk_ids = [

            f"chunk_{i}"

            for i in range(
                len(chunks)
            )

        ]


        # Metadata

        metadatas = [

            {

                "source": chunk["source"],

                "page": chunk["page"]

            }

            for chunk in chunks

        ]


        # Add to ChromaDB

        collection.add(

            ids=chunk_ids,

            documents=chunk_texts,

            embeddings=chunk_embeddings.tolist(),

            metadatas=metadatas

        )


    return collection


try:

    collection = create_collection(

        chunks,

        embedding_model

    )

except Exception as e:

    st.error(
        "Failed to create or load ChromaDB."
    )

    st.exception(e)

    st.stop()


# Show collection information

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

    # Create query embedding

    query_embedding = (

        embedding_model.encode(

            query,

            convert_to_numpy=True

        )

    )


    # Search vector database

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
# 10. LOAD QWEN MODEL
# ============================================================

@st.cache_resource
def load_llm():

    model_name = (
        "Qwen/Qwen2.5-1.5B-Instruct"
    )


    # Select device

    device = (

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )


    # Load tokenizer

    tokenizer = AutoTokenizer.from_pretrained(

        model_name

    )


    # Select data type

    if device == "cuda":

        dtype = torch.float16

    else:

        dtype = torch.float32


    # Load model

    model = AutoModelForCausalLM.from_pretrained(

        model_name,

        dtype=dtype

    )


    # Move model to device

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


# Load LLM only when needed

if "llm_loaded" not in st.session_state:

    st.session_state.llm_loaded = False


# ============================================================
# 11. BUILD CONTEXT
# ============================================================

def build_context(
    results
):

    context_parts = []

    if not results.get(
        "documents"
    ):

        return ""


    if not results["documents"]:

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

    # Retrieve documents

    results = retrieve_documents(

        question,

        top_k=top_k

    )


    # Build context

    context = build_context(
        results
    )


    if not context:

        return (

            "I could not find relevant information "
            "in the provided documents.",

            results

        )


    # Limit context size

    context = context[:6000]


    # Load LLM

    tokenizer, model, device = load_llm()


    # ========================================================
    # Prompt
    # ========================================================

    messages = [

        {

            "role": "system",

            "content": """

You are a technical document question-answering assistant.

Answer the user's question using ONLY the provided context.

Do not use outside knowledge.

If the answer is not available in the context, say:

"I could not find the answer in the provided documents."

Give a clear and concise answer.

"""

        },

        {

            "role": "user",

            "content": f"""

Context:

{context}


Question:

{question}

"""

        }

    ]


    # ========================================================
    # Create prompt
    # ========================================================

    text = tokenizer.apply_chat_template(

        messages,

        tokenize=False,

        add_generation_prompt=True

    )


    # ========================================================
    # Tokenize
    # ========================================================

    inputs = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        max_length=2048

    )


    # Move inputs to device

    inputs = {

        key: value.to(device)

        for key, value in inputs.items()

    }


    # ========================================================
    # Generate answer
    # ========================================================

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=150,

            do_sample=False,

            pad_token_id=tokenizer.eos_token_id

        )


    # ========================================================
    # Extract generated tokens
    # ========================================================

    generated_tokens = outputs[0][

        inputs["input_ids"].shape[1]:

    ]


    # Decode answer

    answer = tokenizer.decode(

        generated_tokens,

        skip_special_tokens=True

    ).strip()


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

                "Searching documents and "
                "generating answer..."

            ):

                answer, results = rag_answer(

                    question,

                    top_k=3

                )


            # =================================================
            # Answer
            # =================================================

            st.subheader(
                "Answer"
            )

            st.write(
                answer
            )


            # =================================================
            # Sources
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

                for metadata in results[

                    "metadatas"

                ][0]:

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
                "An error occurred while "
                "processing your question."
            )

            st.exception(e)


    else:

        st.warning(

            "Please enter a question."

        )
