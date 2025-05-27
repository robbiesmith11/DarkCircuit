import os
import pickle
import hashlib
import asyncio
import threading
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from utils import get_path

_cached_retriever = None
_rag_loading_task = None
_rag_ready = False
_loading_progress = {
    "stage": "Not started",
    "current_file": "",
    "files_processed": 0,
    "total_files": 0,
    "percentage": 0,
    "status": "idle",
    "embedding_progress": 0,
    "total_chunks": 0
}

def get_docs_hash(docs_path):
    """Generate a hash of all PDF files and their modification times."""
    pdf_files = glob(os.path.join(docs_path, "*.pdf"))
    hash_input = ""
    
    for pdf in sorted(pdf_files):
        stat = os.stat(pdf)
        hash_input += f"{pdf}:{stat.st_mtime}:{stat.st_size}"
    
    return hashlib.md5(hash_input.encode()).hexdigest()

def update_progress(stage, current_file="", files_processed=0, total_files=0, embedding_progress=0, total_chunks=0):
    """Update loading progress tracking."""
    global _loading_progress
    _loading_progress["stage"] = stage
    _loading_progress["current_file"] = current_file
    _loading_progress["files_processed"] = files_processed
    _loading_progress["total_files"] = total_files
    _loading_progress["embedding_progress"] = embedding_progress
    _loading_progress["total_chunks"] = total_chunks
    if total_files > 0:
        _loading_progress["percentage"] = int((files_processed / total_files) * 100)
    _loading_progress["status"] = "loading"

def print_progress_bar(current, total, prefix="Progress", length=40):
    """Print a visual progress bar."""
    if total == 0:
        return
    
    percent = int(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix}: |{bar}| {percent}% ({current}/{total})', end='', flush=True)
    if current == total:
        print()  # New line when complete

def load_pdf_parallel(pdf_path, chunk_size, chunk_overlap):
    """Load and process a single PDF in parallel."""
    filename = os.path.basename(pdf_path)
    print(f"📄 Processing: {filename}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    chunks = splitter.split_documents(pages)

    # Attach metadata (e.g. file name) to each chunk.
    for chunk in chunks:
        chunk.metadata["source"] = filename

    return chunks, filename

def load_static_rag_context(docs_path="docs", k=4, chunk_size=1024, chunk_overlap=100):
    """
       Load PDF documents from the given directory, split their content into overlapping
       chunks, embed those chunks into a vector space, and return a FAISS retriever for similarity search.
       
       Now includes persistent caching and parallel processing for improved performance.

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
    print("Loading static RAG context with caching...")
    global _cached_retriever

    update_progress("Initializing", "", 0, 0)
    
    docs_path = get_path(docs_path)
    print(f"📂 Looking for PDFs in: {docs_path}")

    if os.path.isdir(docs_path):
        print("Directory exists. Running Executable")
    else:
        fallback_path = "../docs"
        if os.path.isdir(fallback_path):
            docs_path = fallback_path
            print("Directory does not exist. Running locally in dev mode.")
        else:
            print(f"Neither {docs_path} nor {fallback_path} directories exist.")
            raise FileNotFoundError(f"❌ Neither {docs_path} nor {fallback_path} directories exist.")

    if _cached_retriever is not None:
        return _cached_retriever

    # Create cache directory
    cache_dir = "rag_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate hash of current docs state
    update_progress("Checking cache", "", 0, 0)
    docs_hash = get_docs_hash(docs_path)
    cache_file = os.path.join(cache_dir, f"vectorstore_{docs_hash}.pkl")
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        try:
            update_progress("Loading from cache", "", 0, 0)
            print(f"🚀 Loading cached vectorstore from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            update_progress("Initializing embeddings", "", 0, 0)
            # Recreate embeddings instance (can't be pickled)
            embeddings = FastEmbedEmbeddings()
            
            update_progress("Loading vector store", "", 0, 0)
            # Load FAISS index
            vectorstore = FAISS.load_local(
                cached_data['faiss_path'], 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            
            _cached_retriever = vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": k}
            )
            _loading_progress["status"] = "completed"
            print("✅ Successfully loaded from cache!")
            return _cached_retriever
            
        except Exception as e:
            print(f"⚠️ Cache loading failed: {e}. Rebuilding...")

    # Locate all PDF files in the specified docs_path directory.
    update_progress("Scanning PDFs", "", 0, 0)
    pdf_files = glob(os.path.join(docs_path, "*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"❌ No PDFs found in {docs_path}")

    total_files = len(pdf_files)
    print(f"📚 Processing {total_files} PDFs in parallel...")
    update_progress("Processing PDFs", "", 0, total_files)
    all_documents = []
    files_completed = 0

    # Process PDFs in parallel for better performance
    max_workers = min(4, len(pdf_files))  # Limit to 4 workers to avoid overwhelming system
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all PDF processing tasks
        future_to_pdf = {
            executor.submit(load_pdf_parallel, pdf, chunk_size, chunk_overlap): pdf 
            for pdf in pdf_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_pdf):
            pdf_file = future_to_pdf[future]
            try:
                chunks, filename = future.result()
                all_documents.extend(chunks)
                files_completed += 1
                
                update_progress("Processing PDFs", filename, files_completed, total_files)
                print(f"✅ Processed {filename}: {len(chunks)} chunks ({files_completed}/{total_files})")
            except Exception as e:
                files_completed += 1
                update_progress("Processing PDFs", os.path.basename(pdf_file), files_completed, total_files)
                print(f"❌ Error processing {pdf_file}: {e}")

    print(f"📊 Total documents processed: {len(all_documents)}")

    # Create an embedding function instance.
    update_progress("Downloading embedding models", "", 0, 0)
    print(f"🔄 Generating embeddings for {len(all_documents)} document chunks...")
    print("📦 Note: First run downloads embedding models (~100MB) - this may take several minutes")
    
    embeddings = FastEmbedEmbeddings()
    
    # Build the FAISS vector store from the documents with progress indication
    total_docs = len(all_documents)
    update_progress("Generating embeddings", "", 0, 0, 0, total_docs)
    print(f"⚡ Building vector store for {total_docs} chunks... (this is the slowest step)")
    
    # Process documents in batches to show progress
    batch_size = 100  # Process 100 documents at a time
    vectorstore = None
    
    for i in range(0, total_docs, batch_size):
        batch_end = min(i + batch_size, total_docs)
        batch_docs = all_documents[i:batch_end]
        
        # Update progress with visual bar
        current_progress = i + len(batch_docs)
        update_progress("Generating embeddings", "", 0, 0, current_progress, total_docs)
        print_progress_bar(current_progress, total_docs, "Embedding Progress")
        
        if vectorstore is None:
            # Create initial vectorstore with first batch
            vectorstore = FAISS.from_documents(batch_docs, embeddings)
        else:
            # Add remaining batches to existing vectorstore
            batch_vectorstore = FAISS.from_documents(batch_docs, embeddings)
            vectorstore.merge_from(batch_vectorstore)
    
    print("\n✅ Vector store created successfully!")
    
    # Cache the vectorstore for future use
    try:
        update_progress("Caching vector store", "", 0, 0)
        faiss_cache_path = os.path.join(cache_dir, f"faiss_{docs_hash}")
        vectorstore.save_local(faiss_cache_path)
        
        cache_data = {
            'faiss_path': faiss_cache_path,
            'docs_hash': docs_hash,
            'num_documents': len(all_documents),
            'params': {
                'k': k,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap
            }
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"💾 Cached vectorstore to {cache_file}")
        
        # Clean up old cache files
        cleanup_old_cache(cache_dir, docs_hash)
        
    except Exception as e:
        print(f"⚠️ Failed to cache vectorstore: {e}")

    # Convert the vector store into a retriever with similarity search (using top 'k' results).
    _cached_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    update_progress("Completed", "", total_files, total_files)
    _loading_progress["status"] = "completed"
    print("✅ RAG context loaded successfully!")
    return _cached_retriever

def cleanup_old_cache(cache_dir, current_hash):
    """Remove old cache files to save disk space."""
    try:
        for file in glob(os.path.join(cache_dir, "vectorstore_*.pkl")):
            if current_hash not in file:
                os.remove(file)
                print(f"🗑️ Removed old cache: {file}")
        
        for dir_path in glob(os.path.join(cache_dir, "faiss_*")):
            if current_hash not in dir_path:
                import shutil
                shutil.rmtree(dir_path)
                print(f"🗑️ Removed old FAISS cache: {dir_path}")
                
    except Exception as e:
        print(f"⚠️ Cache cleanup failed: {e}")

def start_rag_background_loading(docs_path="docs", k=4, chunk_size=1024, chunk_overlap=100):
    """Start RAG loading in background thread to avoid blocking app startup."""
    global _rag_loading_task, _rag_ready, _loading_progress
    
    if _rag_loading_task is not None:
        return  # Already started
    
    def background_loader():
        global _cached_retriever, _rag_ready, _loading_progress
        try:
            print("🚀 Starting RAG initialization in background...")
            _loading_progress["status"] = "loading"
            _cached_retriever = load_static_rag_context(docs_path, k, chunk_size, chunk_overlap)
            _rag_ready = True
            print("✅ RAG system ready!")
        except Exception as e:
            print(f"❌ RAG background loading failed: {e}")
            _rag_ready = False
            _loading_progress["status"] = "error"
            _loading_progress["stage"] = f"Error: {str(e)}"
    
    _rag_loading_task = threading.Thread(target=background_loader, daemon=True)
    _rag_loading_task.start()

def get_rag_retriever_async(docs_path="docs", k=4, chunk_size=1024, chunk_overlap=100):
    """Get RAG retriever with async loading support."""
    global _cached_retriever, _rag_ready
    
    if _cached_retriever is not None and _rag_ready:
        return _cached_retriever
    
    if not _rag_ready:
        # Start background loading if not started
        start_rag_background_loading(docs_path, k, chunk_size, chunk_overlap)
        return None  # RAG not ready yet
    
    return _cached_retriever

def is_rag_ready():
    """Check if RAG system is ready for use."""
    return _rag_ready

def get_loading_progress():
    """Get detailed loading progress information."""
    return _loading_progress.copy()