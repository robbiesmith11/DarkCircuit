import os
from glob import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from utils import get_path

_cached_retriever = None

def load_static_rag_context(docs_path="docs", k=4, chunk_size=1024, chunk_overlap=100):
    """
       Load PDF documents from the given directory, split their content into overlapping
       chunks, embed those chunks into a vector space, and return a FAISS retriever for similarity search.

       Parameters:
         docs_path (str): Directory path where the PDFs are stored.
         k (int): Number of top similar documents to retrieve.
         chunk_size (int): Maximum size (in characters) of each chunk.
         chunk_overlap (int): Overlap (in characters) between consecutive chunks.

       Returns:
         A retriever object that can fetch the top 'k' relevant document chunks for a given query.

       Raises:
         FileNotFoundError: When no PDF files are found in the provided directory.
       """
    print("load static rag context")
    global _cached_retriever

    docs_path = get_path(docs_path)
    print(f"📂 Looking for PDFs in: {docs_path}")

    if os.path.isdir(docs_path):
        print("Directory exists. Running Executable")
    else:
        docs_path = "../docs"
        print("Directory does not exist. Running locally in dev mode.")

    if _cached_retriever is not None:
        return _cached_retriever

    # Locate all PDF files in the specified docs_path directory.
    pdf_files = glob(os.path.join(docs_path, "*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"❌ No PDFs found in {docs_path}")

    all_documents = []

    # Load each PDF, split it into pages, further split pages into chunks and add source metadata.
    for pdf in pdf_files:
        print(f"📄 Loading: {pdf}")
        loader = PyPDFLoader(pdf)
        pages = loader.load_and_split()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        chunks = splitter.split_documents(pages)

        # Attach metadata (e.g. file name) to each chunk.
        for chunk in chunks:
            chunk.metadata["source"] = os.path.basename(pdf)

        all_documents.extend(chunks)

    # Create an embedding function instance.
    embeddings = FastEmbedEmbeddings()
    # Build the FAISS vector store from the documents.
    vectorstore = FAISS.from_documents(all_documents, embeddings)
    # Convert the vector store into a retriever with similarity search (using top 'k' results).
    _cached_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    return _cached_retriever
