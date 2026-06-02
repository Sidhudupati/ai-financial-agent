from pypdf import PdfReader

from sentence_transformers import SentenceTransformer

from langchain_text_splitters import RecursiveCharacterTextSplitter

import faiss
import numpy as np


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def extract_pdf_text(uploaded_file):

    try:

        uploaded_file.seek(0)

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    except Exception as e:

        return f"PDF Error: {str(e)}"


def chunk_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    return chunks


def create_embeddings(chunks):

    embeddings = model.encode(chunks)

    return embeddings


def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(embeddings).astype("float32")
    )

    return index


def retrieve_context(
    question,
    chunks,
    index,
    k=3
):

    query_embedding = model.encode(
        [question]
    )

    distances, indices = index.search(
        query_embedding.astype("float32"),
        k
    )

    retrieved_chunks = []

    for idx in indices[0]:

        retrieved_chunks.append(
            chunks[idx]
        )

    return "\n\n".join(
        retrieved_chunks
    )