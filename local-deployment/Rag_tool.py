import os
from glob import glob
from typing import List, Dict, Any, Optional

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

_cached_retriever = None
_cached_vectorstore = None

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
    global _cached_retriever, _cached_vectorstore

    # Return cached retriever if available to avoid redundant loading
    if _cached_retriever is not None:
        return _cached_retriever
        
    docs_path = get_path(docs_path)
    
    if not os.path.isdir(docs_path):
        docs_path = "../docs"

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
    _cached_vectorstore = FAISS.from_documents(all_documents, embeddings)
    # Convert the vector store into a retriever with similarity search (using top 'k' results).
    _cached_retriever = _cached_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    return _cached_retriever

def create_history_aware_rag_retriever(model_name="gpt-3.5-turbo", api_key=None, k=6):
    """
    Create a history-aware retriever that can rewrite queries based on conversation history.
    
    Parameters:
      model_name (str): The OpenAI model to use for query rewriting
      api_key (str): OpenAI API key (optional, defaults to environment variable)
      k (int): Number of documents to retrieve
      
    Returns:
      A history-aware retriever that takes conversation history into account
    """
    global _cached_vectorstore
    
    # Ensure we have a vectorstore loaded
    if _cached_vectorstore is None:
        load_static_rag_context()
        if _cached_vectorstore is None:
            raise ValueError("Failed to initialize vectorstore")
    
    # Create base retriever
    base_retriever = _cached_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    # Create a model for query rewriting - use a lighter model to save tokens/costs
    llm = ChatOpenAI(model=model_name, api_key=api_key)
    
    # Create the history-aware retriever
    retriever = create_history_aware_retriever(
        llm=llm,
        retriever=base_retriever,
        prompt=ChatPromptTemplate.from_messages([
            ("system", """Given a conversation history and the latest user question, 
            formulate a search query that will help retrieve relevant information to 
            answer the user's question. The query should include:
            
            1. Key technical terms from the conversation
            2. Specific HackTheBox or security concepts mentioned
            3. Tools, commands, or techniques discussed
            
            Focus on extracting the underlying information need, not just repeating the question.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
    )
    
    return retriever

def rag_retrieve_with_history(query: str, chat_history: List[Dict[str, Any]], model_name="gpt-3.5-turbo", api_key=None, k=3) -> str:
    """
    Search for relevant documents using a history-aware retriever that
    takes conversation context into account.
    
    Parameters:
      query (str): The user's current query
      chat_history (list): List of past messages (HumanMessage/AIMessage)
      model_name (str): The OpenAI model to use for query rewriting
      api_key (str): OpenAI API key (optional, defaults to environment variable)
      k (int): Number of documents to retrieve
      
    Returns:
      str: Concatenated content from retrieved documents
    """
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
    
    return "\n\n".join(content_parts)
