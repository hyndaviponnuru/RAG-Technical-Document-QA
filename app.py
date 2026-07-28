

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
# 3. DOCUMENT PATH
# ============================================================

PDF_FOLDER = "documents"


# ============================================================
# 4. LOAD PDF DOCUMENTS
# ============================================================

@st.cache_data
def load_documents():

    all_documents = []

    for filename in os.listdir(PDF_FOLDER):

        if filename.endswith(".pdf"):

            file_path = os.path.join(
                PDF_FOLDER,
                filename
            )

            reader = PdfReader(file_path)

            for page_number, page in enumerate(
                reader.pages
            ):

                text = page.extract_text()

                if text:

                    all_documents.append({

                        "text": text,

                        "source": filename,

                        "page": page_number

                    })

    return all_documents


all_documents = load_documents()


# ============================================================
# 5. CHUNK DOCUMENTS
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


# ============================================================
# 6. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# 7. CREATE CHROMADB
# ============================================================

@st.cache_resource
def create_collection():

    client = chromadb.Client()

    collection = client.get_or_create_collection(

        name="technical_documents"

    )

    # Add documents only once

    if collection.count() == 0:

        chunk_texts = [

            chunk["text"]

            for chunk in chunks

        ]

        chunk_embeddings = (

            embedding_model.encode(

                chunk_texts,

                show_progress_bar=True

            )

        )

        chunk_ids = [

            f"chunk_{i}"

            for i in range(
                len(chunks)
            )

        ]

        metadatas = [

            {

                "source":
                chunk["source"],

                "page":
                chunk["page"]

            }

            for chunk in chunks

        ]

        collection.add(

            ids=chunk_ids,

            documents=chunk_texts,

            embeddings=
            chunk_embeddings.tolist(),

            metadatas=metadatas

        )

    return collection


collection = create_collection()


# ============================================================
# 8. RETRIEVAL FUNCTION
# ============================================================

def retrieve_documents(
    query,
    top_k=3
):

    query_embedding = (

        embedding_model.encode(
            query
        )

    )

    results = collection.query(

        query_embeddings=[
            query_embedding.tolist()
        ],

        n_results=top_k

    )

    return results


# ============================================================
# 9. LOAD QWEN MODEL
# ============================================================

@st.cache_resource
def load_llm():

    model_name = (
        "Qwen/Qwen2.5-1.5B-Instruct"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_name
        )
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=(
                torch.float16
                if torch.cuda.is_available()
                else torch.float32
            )
        )
    )

    if torch.cuda.is_available():

        model = model.to("cuda")

    return tokenizer, model


tokenizer, model = load_llm()


# ============================================================
# 10. BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for text in results["documents"][0]:

        context_parts.append(text)

    return "\n\n".join(
        context_parts
    )


# ============================================================
# 11. RAG ANSWER FUNCTION
# ============================================================

def rag_answer(
    question,
    top_k=3
):

    # Retrieve relevant documents

    results = retrieve_documents(

        question,

        top_k=top_k

    )


    # Build context

    context = build_context(
        results
    )

    context = context[:6000]


    # Prompt

    messages = [

        {

            "role": "system",

            "content": """
You are a technical document
question-answering assistant.

Answer the user's question using
ONLY the provided context.

Do not use outside knowledge.

If the answer is not available
in the context, say:

I could not find the answer
in the provided documents.

Give a clear and concise answer
in 2 to 4 sentences.
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


    # Create prompt

    text = tokenizer.apply_chat_template(

        messages,

        tokenize=False,

        add_generation_prompt=True

    )


    # Tokenize

    inputs = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        max_length=4096

    )


    if torch.cuda.is_available():

        inputs = inputs.to("cuda")


    # Generate

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=150,

            do_sample=False

        )


    # Extract answer

    generated_tokens = outputs[0][

        inputs.input_ids.shape[1]:

    ]


    answer = tokenizer.decode(

        generated_tokens,

        skip_special_tokens=True

    ).strip()


    return answer, results


# ============================================================
# 12. STREAMLIT USER INTERFACE
# ============================================================

question = st.text_input(

    "Ask your question",

    placeholder=
    "Example: What is Retrieval-Augmented Generation?"

)


if st.button("🔍 Ask Question"):

    if question.strip():

        with st.spinner(

            "Searching documents and generating answer..."

        ):

            answer, results = rag_answer(

                question,

                top_k=3

            )


        # Answer

        st.subheader(
            "Answer"
        )

        st.write(
            answer
        )


        # Sources

        st.subheader(
            "Sources"
        )


        shown_sources = set()


        for i in range(

            len(
                results["documents"][0]
            )

        ):

            source = results[
                "metadatas"
            ][0][i]["source"]


            page = results[
                "metadatas"
            ][0][i]["page"]


            citation = (

                source,

                page

            )


            if citation not in shown_sources:

                st.write(

                    f"📄 {source} | Page {page}"

                )

                shown_sources.add(

                    citation

                )

    else:

        st.warning(

            "Please enter a question."

        )
