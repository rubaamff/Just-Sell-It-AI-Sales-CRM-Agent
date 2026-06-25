
import os
# macOS OpenMP workaround — must be set before FAISS / numpy import.
# See top of app.py for context. Setting here too so any code path
# that imports utils.rag directly is protected.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"

def initialize_rag():
    """
    Loads all PDF files from the data/ folder and builds a FAISS vector store.
    Returns a retriever ready for querying.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        raise FileNotFoundError(
            f"The '{DATA_DIR}/' folder was just created. "
            "Please add your BeamData PDF files (e.g. 'Beamdata Past Project Descriptions.pdf' "
            "and 'Beam Data AI Hub Intro.pdf') into the data/ folder, then try again."
        )

    pdf_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{DATA_DIR}/' folder. "
            "Please add your BeamData PDF files there and try again."
        )

    all_documents = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        all_documents.extend(docs)
        print(f"Loaded: {pdf_path} ({len(docs)} pages)")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"Total chunks created: {len(chunks)}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.6, "k": 5},
    )
    return retriever