"""
Enhanced RAG System for DarkCircuit Agent

This module provides sophisticated retrieval capabilities including:
- Hybrid semantic and keyword-based search
- Metadata filtering and query expansion
- Conversation-aware document retrieval
- Document re-ranking based on relevance
"""

import os
import re
import time
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from datetime import datetime
import numpy as np
from collections import Counter

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Try to import advanced_memory if available
try:
    from advanced_memory import AdvancedMemory, convert_to_memory_format
    ADVANCED_MEMORY_AVAILABLE = True
except ImportError:
    ADVANCED_MEMORY_AVAILABLE = False

# Cache for vector stores and retrievers
_cached_vector_stores = {}
_cached_retrievers = {}
_cached_bm25_indices = {}  # For keyword search

class BM25Index:
    """Simple BM25 implementation for keyword search."""
    
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1  # term frequency normalization
        self.b = b    # document length normalization
        self.doc_freqs = Counter()  # df values for each term
        self.doc_lengths = []  # length of each document
        self.avg_doc_length = 0  # average document length
        self.docs = []  # document texts
        self.vocab = set()  # vocabulary set
        self.idf = {}  # idf values
        self.doc_vectors = []  # pre-computed document vectors
        self.metadata = []  # metadata for each document
        
    def tokenize(self, text):
        """Simple tokenization by splitting on non-alphanumeric characters."""
        return re.findall(r'\w+', text.lower())
    
    def fit(self, documents: List[Document]):
        """
        Process documents to build the BM25 index.
        
        Args:
            documents: List of Document objects with page_content and metadata
        """
        self.docs = [doc.page_content for doc in documents]
        self.metadata = [doc.metadata for doc in documents]
        
        # Calculate document frequencies and lengths
        for doc in self.docs:
            tokens = self.tokenize(doc)
            self.doc_lengths.append(len(tokens))
            
            # Count document frequency for each term
            term_set = set(tokens)
            for term in term_set:
                self.doc_freqs[term] += 1
                self.vocab.add(term)
        
        # Calculate average document length and IDF values
        self.avg_doc_length = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        num_docs = len(self.docs)
        
        # Calculate IDF values
        for term in self.vocab:
            self.idf[term] = np.log((num_docs - self.doc_freqs[term] + 0.5) / 
                                  (self.doc_freqs[term] + 0.5) + 1.0)
        
        # Pre-compute document vectors for faster retrieval
        self._precompute_doc_vectors()
    
    def _precompute_doc_vectors(self):
        """Pre-compute document vectors for faster scoring."""
        self.doc_vectors = []
        for i, doc in enumerate(self.docs):
            doc_vector = {}
            tokens = self.tokenize(doc)
            term_freqs = Counter(tokens)
            doc_len = self.doc_lengths[i]
            
            for term, freq in term_freqs.items():
                # TF component: (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * doc_len / self.avg_doc_length))
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                tf = numerator / denominator
                # Store the product of TF * IDF
                if term in self.idf:
                    doc_vector[term] = tf * self.idf.get(term, 0)
            
            self.doc_vectors.append(doc_vector)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, Document]]:
        """
        Search for top k documents matching the query.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, score, Document) tuples
        """
        query_terms = self.tokenize(query)
        scores = []
        
        # Calculate scores for each document
        for i, doc_vector in enumerate(self.doc_vectors):
            score = 0.0
            for term in query_terms:
                if term in doc_vector:
                    score += doc_vector[term]
            
            scores.append((i, score))
        
        # Sort by score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for i, score in scores[:top_k]:
            doc = Document(page_content=self.docs[i], metadata=self.metadata[i])
            results.append((i, score, doc))
            
        return results


class HybridRetriever:
    """
    Hybrid retriever combining semantic search and keyword-based search.
    Supports both FAISS (vector) and BM25 (keyword) retrieval.
    """
    
    def __init__(self, 
                 vector_store: FAISS,
                 bm25_index: Optional[BM25Index] = None,
                 docs_path: str = "docs",
                 vector_weight: float = 0.7,
                 rerank: bool = True,
                 llm: Optional[Any] = None):
        """
        Initialize the hybrid retriever.
        
        Args:
            vector_store: FAISS vector store for semantic search
            bm25_index: Optional BM25 index for keyword search
            docs_path: Path to the documents directory
            vector_weight: Weight for vector similarity (vs keyword search)
            rerank: Whether to rerank results
            llm: Optional language model for query expansion and reranking
        """
        self.vector_store = vector_store
        self.vector_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # Retrieve more to allow for reranking
        )
        
        self.bm25_index = bm25_index
        self.docs_path = docs_path
        self.vector_weight = vector_weight
        self.rerank = rerank
        self.llm = llm

        # Optional entities and conversation memory
        self.memory = None  # Will hold AdvancedMemory if available
        
    def set_memory(self, memory: Any):
        """Set the memory object for context-aware retrieval."""
        self.memory = memory
    
    def expand_query(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Expand the query using the language model for better retrieval.
        
        Args:
            query: Original query string
            chat_history: Optional chat history for context
            
        Returns:
            Expanded query
        """
        if not self.llm:
            return query
            
        try:
            # Context from recent entities if memory is available
            context_str = ""
            if self.memory and hasattr(self.memory, 'get_entities'):
                entities = self.memory.get_entities()
                if entities:
                    context_str += "Important entities from your conversation:\n"
                    for e in entities[:5]:  # Limit to 5 entities
                        context_str += f"- {e['name']} ({e['entity_type']}): {e['value']}\n"
            
            # Context from chat history
            if chat_history and len(chat_history) > 0:
                context_str += "\nRecent conversation:\n"
                for i, msg in enumerate(chat_history[-3:]):  # Last 3 messages
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    context_str += f"{role}: {content[:100]}...\n"
            
            # Only proceed with expansion if we have context
            if not context_str:
                return query
                
            # Create system prompt for query expansion
            system_prompt = f"""
            You are an expert security researcher helping with a query expansion task.
            Given a user's search query and conversation context, rewrite the query to be more specific and include relevant technical terms.
            Focus on cybersecurity terminology that would help retrieve useful documents.
            Don't change the intent of the query, just make it more precise and include additional relevant terms.
            
            Context information:
            {context_str}
            """
            
            # Query the LLM for expansion
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Original query: {query}\n\nExpanded query:"}
            ]
            
            response = self.llm.invoke(messages)
            expanded_query = response.content.strip()
            
            print(f"Expanded query: {expanded_query}")
            return expanded_query
            
        except Exception as e:
            print(f"Query expansion error: {e}")
            return query  # Fallback to original query
    
    def retrieve_with_context(self, 
                             query: str, 
                             chat_history: Optional[List[Dict[str, Any]]] = None, 
                             metadata_filter: Optional[Dict[str, Any]] = None,
                             top_k: int = 6) -> List[Document]:
        """
        Retrieve documents using both semantic and keyword search, with context awareness.
        
        Args:
            query: Query string
            chat_history: Optional chat history for context
            metadata_filter: Optional metadata constraints
            top_k: Number of documents to return
            
        Returns:
            List of relevant documents
        """
        # If memory system is available, try to get context from it
        memory_context = None
        if ADVANCED_MEMORY_AVAILABLE and self.memory:
            # Convert chat history to memory format if needed
            if chat_history and not isinstance(chat_history[0], dict):
                chat_history = convert_to_memory_format(chat_history)
                
            # Add chat history to memory if available
            if chat_history:
                for msg in chat_history:
                    self.memory.add_message(msg)
                    
            # Get context for query
            memory_context = self.memory.get_context_for_query(query)
            
            # Expand query with memory context
            if memory_context:
                expanded_query = self.expand_query(query, chat_history)
                print(f"Using memory context: {memory_context[:100]}...")
            else:
                expanded_query = self.expand_query(query, chat_history)
        else:
            # Basic query expansion without memory
            expanded_query = self.expand_query(query, chat_history)
                
        # Semantic search (vector)
        vector_docs = self.vector_retriever.get_relevant_documents(expanded_query)
        
        # Keyword search (BM25) if available
        keyword_docs = []
        if self.bm25_index:
            keyword_results = self.bm25_index.search(expanded_query, top_k=top_k)
            keyword_docs = [doc for _, _, doc in keyword_results]
        
        # Apply metadata filtering if specified
        if metadata_filter:
            if vector_docs:
                vector_docs = [doc for doc in vector_docs 
                              if all(doc.metadata.get(k) == v for k, v in metadata_filter.items())]
            if keyword_docs:
                keyword_docs = [doc for doc in keyword_docs 
                               if all(doc.metadata.get(k) == v for k, v in metadata_filter.items())]
        
        # Combine results with deduplication
        seen_content = set()
        hybrid_results = []
        
        # Weight vector results higher by adding them first
        for doc in vector_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                hybrid_results.append((doc, self.vector_weight))
        
        # Add keyword results with lower weight
        for doc in keyword_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                hybrid_results.append((doc, 1.0 - self.vector_weight))
                
        # Rerank the combined results if enabled
        if self.rerank and self.llm and len(hybrid_results) > 1:
            reranked_results = self._rerank_results(query, expanded_query, hybrid_results)
            final_docs = [doc for doc, _ in reranked_results[:top_k]]
        else:
            # Sort by weight and take top k
            hybrid_results.sort(key=lambda x: x[1], reverse=True)
            final_docs = [doc for doc, _ in hybrid_results[:top_k]]
        
        return final_docs
    
    def _rerank_results(self, 
                        original_query: str, 
                        expanded_query: str, 
                        results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """
        Rerank results based on relevance to query.
        
        Args:
            original_query: The original user query
            expanded_query: The expanded query used for retrieval
            results: List of (Document, score) tuples
            
        Returns:
            Reranked list of (Document, score) tuples
        """
        if not results:
            return []
            
        try:
            # Simple reranking without LLM by combining keywords from both queries
            query_terms = set(re.findall(r'\w+', original_query.lower()))
            expanded_terms = set(re.findall(r'\w+', expanded_query.lower()))
            all_terms = query_terms.union(expanded_terms)
            
            reranked = []
            for doc, initial_score in results:
                # Calculate term overlap
                doc_terms = set(re.findall(r'\w+', doc.page_content.lower()))
                overlap = len(all_terms.intersection(doc_terms))
                term_score = overlap / max(1, len(all_terms))
                
                # Calculate density of query terms
                term_density = 0
                for term in all_terms:
                    term_count = doc.page_content.lower().count(term)
                    term_density += term_count
                
                # Normalize by document length
                density_score = term_density / max(1, len(doc.page_content.split()))
                
                # Combine scores
                final_score = 0.4 * initial_score + 0.4 * term_score + 0.2 * density_score
                reranked.append((doc, final_score))
                
            # Sort by combined score
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked
            
        except Exception as e:
            print(f"Reranking error: {e}")
            return results  # Return original results on error
    
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Standard interface method for compatibility with LangChain.
        
        Args:
            query: Query string
            **kwargs: Additional parameters
            
        Returns:
            List of relevant documents
        """
        chat_history = kwargs.get("chat_history", [])
        metadata_filter = kwargs.get("metadata_filter", None)
        top_k = kwargs.get("k", 6)
        
        return self.retrieve_with_context(query, chat_history, metadata_filter, top_k)
    
    def invoke(self, input_dict: Dict[str, Any]) -> List[Document]:
        """
        Invoke interface for compatibility with LangChain.
        
        Args:
            input_dict: Dictionary with query and optional parameters
            
        Returns:
            List of relevant documents
        """
        query = input_dict.get("input", "")
        chat_history = input_dict.get("chat_history", [])
        metadata_filter = input_dict.get("metadata_filter", None)
        top_k = input_dict.get("k", 6)
        
        return self.retrieve_with_context(query, chat_history, metadata_filter, top_k)


def load_static_rag_context(docs_path="docs", 
                          k=3,  # Reduced from 6
                          chunk_size=512,  # Reduced from 1024
                          chunk_overlap=50,  # Reduced from 100
                          use_bm25=True,
                          hybrid_retrieval=True,
                          memory: Optional[Any] = None,
                          use_llm_reranking=False,
                          api_key=None):
    """
    Enhanced version of load_static_rag_context with advanced retrieval options.
    
    Args:
        docs_path: Directory path where the PDFs are stored
        k: Number of documents to retrieve
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        use_bm25: Whether to use BM25 for keyword search
        hybrid_retrieval: Whether to use hybrid retrieval
        memory: Optional AdvancedMemory instance
        use_llm_reranking: Whether to use LLM-based reranking
        api_key: Optional OpenAI API key
        
    Returns:
        A retriever that can fetch relevant document chunks
    """
    global _cached_vector_stores, _cached_retrievers, _cached_bm25_indices
    
    # Generate a unique cache key based on parameters
    cache_key = f"{docs_path}_{chunk_size}_{chunk_overlap}"
    
    if cache_key in _cached_retrievers:
        retriever = _cached_retrievers[cache_key]
        # Update memory if provided
        if memory is not None and hasattr(retriever, 'set_memory'):
            retriever.set_memory(memory)
        return retriever
    
    # Check if path exists and adjust if needed
    if not os.path.isdir(docs_path):
        if os.path.isdir("../docs"):
            docs_path = "../docs"
        else:
            raise FileNotFoundError(f"No documents directory found at {docs_path}")
    
    # Load documents if not cached
    if cache_key not in _cached_vector_stores:
        # Find PDF files
        pdf_files = []
        for root, _, files in os.walk(docs_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDFs found in {docs_path}")
            
        # Load and process documents
        all_documents = []
        
        for pdf in pdf_files:
            loader = PyPDFLoader(pdf)
            pages = loader.load_and_split()
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True
            )
            chunks = splitter.split_documents(pages)
            
            # Add source and chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata["source"] = os.path.basename(pdf)
                chunk.metadata["chunk_id"] = f"{os.path.basename(pdf)}_chunk_{i}"
                chunk.metadata["file_path"] = pdf
            
            all_documents.extend(chunks)
        
        # Create embedding function and vector store
        embeddings = FastEmbedEmbeddings()
        vector_store = FAISS.from_documents(all_documents, embeddings)
        _cached_vector_stores[cache_key] = vector_store
        
        # Create BM25 index if requested
        if use_bm25:
            bm25_index = BM25Index()
            bm25_index.fit(all_documents)
            _cached_bm25_indices[cache_key] = bm25_index
        else:
            bm25_index = None
    else:
        # Use cached instances
        vector_store = _cached_vector_stores[cache_key]
        bm25_index = _cached_bm25_indices.get(cache_key)

    # Create a language model for advanced features if requested
    llm = None
    if use_llm_reranking:
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
        except Exception as e:
            print(f"Error creating LLM for retrieval: {e}")
    
    # Create and return the appropriate retriever
    if hybrid_retrieval:
        # Create hybrid retriever
        retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            docs_path=docs_path,
            vector_weight=0.7,
            rerank=True,
            llm=llm
        )
        
        # Set memory if provided
        if memory is not None:
            retriever.set_memory(memory)
    else:
        # Standard vector retriever
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": k}
        )
    
    # Cache the retriever
    _cached_retrievers[cache_key] = retriever
    
    return retriever


def rag_retrieve_with_history(query: str, 
                             chat_history: List[Dict[str, Any]], 
                             model_name="gpt-3.5-turbo", 
                             api_key=None, 
                             k=6, 
                             memory=None) -> str:
    """
    Enhanced context-aware document retrieval that takes conversation history into account.
    
    Args:
        query: The user's current query
        chat_history: List of past messages (HumanMessage/AIMessage or dicts)
        model_name: OpenAI model for query reformulation
        api_key: Optional OpenAI API key
        k: Number of documents to retrieve
        memory: Optional AdvancedMemory instance
        
    Returns:
        Concatenated content from retrieved documents
    """
    # Convert dictionary messages to proper LangChain format
    formatted_history = []
    
    for msg in chat_history:
        if isinstance(msg, dict):
            if msg.get("type") == "human" or msg.get("role") == "user":
                formatted_history.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("type") == "ai" or msg.get("role") == "assistant":
                formatted_history.append(AIMessage(content=msg.get("content", "")))
        elif isinstance(msg, (HumanMessage, AIMessage)):
            formatted_history.append(msg)
    
    # Create a hybrid retriever with memory awareness
    retriever = load_static_rag_context(
        k=k,
        use_bm25=True,
        hybrid_retrieval=True,
        memory=memory,
        use_llm_reranking=True,
        api_key=api_key
    )
    
    # Retrieve documents using conversation context
    docs = retriever.invoke({
        "input": query, 
        "chat_history": formatted_history
    })
    
    # Format the results
    content_parts = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata
        source = metadata.get("source", "Unknown source")
        content_parts.append(f"[Source {i + 1}: {source}] {doc.page_content}")
    
    return "\n\n".join(content_parts)


# Create a conversation-aware RAG chain
def create_conversation_aware_rag_chain(llm, retriever):
    """
    Create a conversation-aware RAG chain that uses chat history for better retrieval.
    
    Args:
        llm: Language model
        retriever: Document retriever
        
    Returns:
        RAG chain that incorporates conversation history
    """
    # System prompt for the RAG chain
    system_prompt = """You are a security assistant retrieving information to help with cybersecurity challenges.
    Consider the conversation history and the retrieved document snippets to provide a comprehensive answer.
    
    When information from the documents conflicts with the conversation history, prioritize:
    1. The most recent messages in the conversation
    2. The most relevant documents for the specific question
    3. Documents that provide technical details over general information
    
    Format your response to be readable with paragraphs, bullet points for lists, and code blocks for commands or code.
    """
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("system", "Here are relevant documents to help answer the query:\n\n{context}")
    ])
    
    # Create and return the chain
    chain = prompt | llm | StrOutputParser()
    
    # Create a retrieval wrapper function
    def retrieve_and_generate(input_dict):
        input_query = input_dict["input"]
        chat_history = input_dict.get("chat_history", [])
        
        # Get relevant documents
        docs = retriever.invoke({
            "input": input_query,
            "chat_history": chat_history
        })
        
        # Join document content
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response
        return chain.invoke({
            "input": input_query,
            "chat_history": chat_history,
            "context": context
        })
    
    return retrieve_and_generate