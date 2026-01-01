from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

VECTOR_DB_PATH = "ielts_vectordb"

def build_vectordb(pdf_path="data/ielts_band_descriptors.pdf"):
    if os.path.exists(VECTOR_DB_PATH):
        embeddings = HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(VECTOR_DB_PATH)
    return vectordb

def retrieve(vectordb, essay, criterion):
    query = f"{criterion} IELTS band descriptor guidance"
    return vectordb.similarity_search(query, k=3)
