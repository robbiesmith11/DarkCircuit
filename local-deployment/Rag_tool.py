import os
import pickle
import hashlib
from glob import glob
from typing import List, Dict, Any, Optional
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from utils import get_path

# Global variables for persistent caching throughout application lifecycle
_cached_retriever = None
_cached_vectorstore = None
_cached_history_retriever = None
_query_cache = {}  # Cache for common queries
_rag_initialized = False  # Track if RAG has been initialized this session

def load_static_rag_context(docs_path="docs", k=4, chunk_size=1024, chunk_overlap=100):
    """
       Load PDF documents from the given directory, split their content into overlapping
       chunks, embed those chunks into a vector space, and return a FAISS retriever for similarity search.
       
       OPTIMIZED: Uses persistent caching to avoid rebuilding vectorstore on each startup.
       Implements lazy loading that initializes once per application session.

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
    global _cached_retriever, _cached_vectorstore, _rag_initialized

    # Return cached retriever if already initialized this session
    if _rag_initialized and _cached_retriever is not None:
        print("🚀 Using in-memory cached RAG retriever (already initialized this session)")
        return _cached_retriever
        
    docs_path = get_path(docs_path)
    
    if not os.path.isdir(docs_path):
        docs_path = "../docs"

    # Locate all PDF files in the specified docs_path directory.
    pdf_files = glob(os.path.join(docs_path, "*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"❌ No PDFs found in {docs_path}")

    # Create cache directory
    cache_dir = get_path("rag_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a hash of PDF files and config for cache key
    pdf_hash = hashlib.md5(
        "".join(sorted(pdf_files)).encode() + 
        f"{chunk_size}_{chunk_overlap}_{k}".encode()
    ).hexdigest()
    
    cache_file = os.path.join(cache_dir, f"vectorstore_{pdf_hash}.pkl")
    
    # Try to load from cache first
    try:
        if os.path.exists(cache_file):
            print(f"🚀 Loading cached vectorstore from {cache_file}")
            start_time = time.time()
            with open(cache_file, 'rb') as f:
                _cached_vectorstore = pickle.load(f)
            _cached_retriever = _cached_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
            
            # Mark as initialized for this application session
            _rag_initialized = True
            
            print(f"✅ Loaded vectorstore from cache in {time.time() - start_time:.2f}s - RAG now persistent for this session")
            return _cached_retriever
    except Exception as e:
        print(f"⚠️ Failed to load from cache: {e}")

    print(f"🔄 Building new vectorstore (this may take a moment...)")
    start_time = time.time()
    
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
    _cached_vectorstore = FAISS.from_documents(all_documents, embeddings)
    
    # Save to cache for future use
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(_cached_vectorstore, f)
        print(f"💾 Saved vectorstore to cache")
    except Exception as e:
        print(f"⚠️ Failed to save to cache: {e}")
    
    # Convert the vector store into a retriever with similarity search (using top 'k' results).
    _cached_retriever = _cached_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    # Mark as initialized for this application session
    _rag_initialized = True
    
    print(f"✅ Built vectorstore in {time.time() - start_time:.2f}s - RAG now persistent for this session")
    return _cached_retriever

def create_history_aware_rag_retriever(model_name="gpt-4o-mini", api_key=None, k=4):
    """
    Create a history-aware retriever that can rewrite queries based on conversation history.
    
    OPTIMIZED: Uses lighter model and caches retriever instance throughout application session.
    
    Parameters:
      model_name (str): The OpenAI model to use for query rewriting
      api_key (str): OpenAI API key (optional, defaults to environment variable)
      k (int): Number of documents to retrieve
      
    Returns:
      A history-aware retriever that takes conversation history into account
    """
    global _cached_vectorstore, _cached_history_retriever, _rag_initialized
    
    # Return cached history retriever if available and RAG is initialized
    if _rag_initialized and _cached_history_retriever is not None:
        return _cached_history_retriever
    
    # Ensure we have a vectorstore loaded
    if _cached_vectorstore is None:
        load_static_rag_context()
        if _cached_vectorstore is None:
            raise ValueError("Failed to initialize vectorstore")
    
    # Create base retriever with fewer documents for speed
    base_retriever = _cached_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    # Create a lighter model for query rewriting to save time and tokens
    llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0)
    
    # Create the history-aware retriever with a simpler prompt
    _cached_history_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=base_retriever,
        prompt=ChatPromptTemplate.from_messages([
            ("system", "Rewrite the query using context from conversation history. Focus on HTB/security terms."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
    )
    
    return _cached_history_retriever

def rag_retrieve_with_history(query: str, chat_history: List[Dict[str, Any]], model_name="gpt-4o-mini", api_key=None, k=3) -> str:
    """
    Search for relevant documents using a history-aware retriever that
    takes conversation context into account.
    
    OPTIMIZED: Uses query caching and lighter model.
    
    Parameters:
      query (str): The user's current query
      chat_history (list): List of past messages (HumanMessage/AIMessage)
      model_name (str): The OpenAI model to use for query rewriting
      api_key (str): OpenAI API key (optional, defaults to environment variable)
      k (int): Number of documents to retrieve
      
    Returns:
      str: Concatenated content from retrieved documents
    """
    global _query_cache
    
    # Create cache key from query and recent history
    history_str = ""
    if chat_history and len(chat_history) > 0:
        # Only use last 2 messages for cache key to avoid cache misses
        recent_history = chat_history[-2:] if len(chat_history) > 2 else chat_history
        history_str = str([msg.get("content", "") if isinstance(msg, dict) else str(msg) for msg in recent_history])
    
    cache_key = hashlib.md5(f"{query}_{history_str}_{k}".encode()).hexdigest()
    
    # Check cache first
    if cache_key in _query_cache:
        print(f"🚀 Using cached RAG result for query")
        return _query_cache[cache_key]
    
    start_time = time.time()
    
    # Convert dictionary messages to proper LangChain message objects if needed
    formatted_history = []
    for msg in chat_history:
        if isinstance(msg, dict):
            if msg.get("type") == "human" or msg.get("role") == "user":
                formatted_history.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("type") == "ai" or msg.get("role") == "assistant":
                formatted_history.append(AIMessage(content=msg.get("content", "")))
        elif isinstance(msg, (HumanMessage, AIMessage)):
            formatted_history.append(msg)
    
    # Create the history-aware retriever
    retriever = create_history_aware_rag_retriever(model_name=model_name, api_key=api_key, k=k)
    
    # Retrieve relevant documents
    docs = retriever.invoke({"input": query, "chat_history": formatted_history})
    
    # Format the results
    content_parts = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata
        source = metadata.get("source", "Unknown source")
        content_parts.append(f"[Source {i + 1}: {source}] {doc.page_content}")
    
    result = "\n\n".join(content_parts)
    
    # Cache the result (limit cache size to prevent memory issues)
    if len(_query_cache) < 50:  # Limit cache to 50 entries
        _query_cache[cache_key] = result
    elif len(_query_cache) >= 50:
        # Clear cache when it gets too full
        _query_cache.clear()
        _query_cache[cache_key] = result
    
    print(f"✅ RAG retrieval completed in {time.time() - start_time:.2f}s")
    return result

def preload_rag_system():
    """
    Preload the RAG system during application startup to avoid delays on first use.
    This ensures the vectorstore is loaded into memory and ready for immediate use.
    """
    global _rag_initialized
    
    if _rag_initialized:
        print("🚀 RAG system already initialized")
        return
        
    print("🔄 Preloading RAG system for faster response times...")
    try:
        # Load the vectorstore and mark as initialized
        load_static_rag_context()
        print("✅ RAG system preloaded successfully - ready for instant responses")
    except Exception as e:
        print(f"⚠️ Failed to preload RAG system: {e}")

def is_rag_initialized():
    """
    Check if the RAG system has been initialized this session.
    
    Returns:
        bool: True if RAG is initialized and ready to use
    """
    return _rag_initialized
