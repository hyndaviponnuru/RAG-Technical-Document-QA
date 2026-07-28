import os
import streamlit as st
import chromadb

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# ==========================================
# STREAMLIT PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Technical Document QA Assistant",
    page_icon="📚",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("📚 Technical Document QA Assistant")

st.write(
    "Ask questions about technical documents "
    "using Retrieval-Augmented Generation (RAG)."
)


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ==========================================
# LOAD QWEN MODEL
# ==========================================

@st.cache_resource
def load_llm():

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )

    return tokenizer, model


tokenizer, model = load_llm()


# ==========================================
# DOCUMENT PATH
# ==========================================

DOCUMENT_PATH = "documents"


# ==========================================
# LOAD PDF DOCUMENTS
# ==========================================

@st.cache_data
def load_documents():

    documents = []

    for filename in os.listdir(
        DOCUMENT_PATH
    ):

        if filename.endswith(".pdf"):

            filepath = os.path.join(
                DOCUMENT_PATH,
                filename
            )

            reader = PdfReader(
                filepath
            )

            for page_number, page in enumerate(
                reader.pages
            ):

                text = page.extract_text()

                if text:

                    documents.append({

                        "text": text,

                        "source": filename,

                        "page": page_number

                    })

    return documents


documents = load_documents()


# ==========================================
# CREATE CHUNKS
# ==========================================

def create_chunks(
    documents,
    chunk_size=500
):

    chunks = []

    for doc in documents:

        text = doc["text"]

        for i in range(
            0,
            len(text),
            chunk_size
        ):

            chunk_text = text[
                i:i + chunk_size
            ]

            chunks.append({

                "text": chunk_text,

                "source": doc["source"],

                "page": doc["page"]

            })

    return chunks


chunks = create_chunks(
    documents
)


# ==========================================
# CREATE CHROMADB
# ==========================================

@st.cache_resource
def create_vector_database():

    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="technical_documents"
    )

    if collection.count() == 0:

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = embedding_model.encode(
            texts
        ).tolist()

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

            documents=texts,

            embeddings=embeddings,

            metadatas=metadatas

        )

    return collection


collection = create_vector_database()


# ==========================================
# RETRIEVAL FUNCTION
# ==========================================

def retrieve_documents(
    question,
    top_k=3
):

    question_embedding = (
        embedding_model.encode(
            [question]
        ).tolist()
    )

    results = collection.query(

        query_embeddings=
        question_embedding,

        n_results=top_k

    )

    return results


# ==========================================
# RAG ANSWER FUNCTION
# ==========================================

def rag_answer(
    question
):

    results = retrieve_documents(
        question,
        top_k=3
    )

    context = "\n\n".join(

        results["documents"][0]

    )

    context = context[:6000]


    messages = [

        {
            "role": "system",

            "content": """
You are a technical document
question-answering assistant.

Answer ONLY using the provided context.

If the answer is not available
in the context, say:

I could not find the answer
in the provided documents.

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


    text = tokenizer.apply_chat_template(

        messages,

        tokenize=False,

        add_generation_prompt=True

    )


    inputs = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        max_length=4096

    )


    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=150,

            do_sample=False

        )


    generated_tokens = outputs[0][

        inputs.input_ids.shape[1]:

    ]


    answer = tokenizer.decode(

        generated_tokens,

        skip_special_tokens=True

    ).strip()


    return answer, results


# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================

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
                question
            )


        st.subheader(
            "Answer"
        )

        st.write(
            answer
        )


        st.subheader(
            "Sources"
        )


        shown_sources = set()


        for i in range(3):

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

                    f"📄 {source} — Page {page}"

                )

                shown_sources.add(
                    citation
                )

    else:

        st.warning(
            "Please enter a question."
        )
