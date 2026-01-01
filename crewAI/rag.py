from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

VECTOR_DB_PATH = "ielts_vectordb"
PDF_PATH = "data/ielts_band_descriptors.pdf"

def build_vectordb():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)

    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(VECTOR_DB_PATH)
    return vectordb

def retrieve(vectordb, criterion, k=3):
    query = f"IELTS {criterion} band descriptors"
    return vectordb.similarity_search(query, k=k)
