"""
Context Integration Module for DarkCircuit Agent

This module provides a unified interface to integrate all context-aware components:
- Advanced memory system
- Enhanced RAG with hybrid retrieval
- Dynamic prompt optimization
- Task tracking and agent state
- Context persistence

This creates a complete context awareness system that can be easily integrated
with the DarkCircuit agent to dramatically improve its context retention abilities.
"""

import os
import time
import json
import uuid
import pickle
from typing import List, Dict, Any, Optional, Tuple, Set, Union, Callable
from datetime import datetime
from collections import deque

# Import all component modules
try:
    from advanced_memory import (
        AdvancedMemory, 
        MemoryItem, 
        MessageMemoryItem, 
        Entity, 
        Conversation,
        create_advanced_memory
    )
    ADVANCED_MEMORY_AVAILABLE = True
except ImportError:
    ADVANCED_MEMORY_AVAILABLE = False

try:
    from enhanced_rag import (
        HybridRetriever,
        load_static_rag_context,
        rag_retrieve_with_history,
        create_conversation_aware_rag_chain
    )
    ENHANCED_RAG_AVAILABLE = True
except ImportError:
    ENHANCED_RAG_AVAILABLE = False

try:
    from dynamic_prompting import (
        DynamicPromptGenerator,
        PromptLibrary,
        create_prompt_generator
    )
    DYNAMIC_PROMPTING_AVAILABLE = True
except ImportError:
    DYNAMIC_PROMPTING_AVAILABLE = False

try:
    from task_tracking import (
        TaskTracker,
        TaskStatus,
        TaskType,
        TaskPriority,
        create_task_tracker
    )
    TASK_TRACKING_AVAILABLE = True
except ImportError:
    TASK_TRACKING_AVAILABLE = False


class ContextManager:
    """
    Unified manager for all context-aware components of the DarkCircuit agent.
    Integrates memory, RAG, prompting, and task tracking into a cohesive system.
    """
    
    def __init__(self, 
                 session_id: Optional[str] = None,
                 persistence_path: Optional[str] = "./context_data",
                 enable_memory: bool = True,
                 enable_rag: bool = True,
                 enable_dynamic_prompts: bool = True,
                 enable_task_tracking: bool = True,
                 default_reasoner_prompt: Optional[str] = None,
                 default_responder_prompt: Optional[str] = None,
                 memory_params: Optional[Dict[str, Any]] = None,
                 rag_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the context manager.
        
        Args:
            session_id: Optional unique session identifier
            persistence_path: Path for persisting context data
            enable_memory: Whether to enable advanced memory
            enable_rag: Whether to enable enhanced RAG
            enable_dynamic_prompts: Whether to enable dynamic prompts
            enable_task_tracking: Whether to enable task tracking
            default_reasoner_prompt: Default prompt for reasoner
            default_responder_prompt: Default prompt for responder
            memory_params: Optional parameters for memory system
            rag_params: Optional parameters for RAG system
        """
        # Generate a session ID if not provided
        self.session_id = session_id or str(uuid.uuid4())[:8]
        
        # Ensure persistence directory exists
        self.persistence_path = persistence_path
        if persistence_path and not os.path.exists(persistence_path):
            os.makedirs(persistence_path, exist_ok=True)
            
        # System state
        self.created_at = time.time()
        self.components_initialized = set()
        self.last_query = None
        self.last_response = None
        self.conversation_active = True
        
        # Command tracking to avoid repetition but ensure comprehensive execution
        self.executed_commands = set()
        self.command_results = {}
        self.command_timestamps = []  # Ordered list of command execution timestamps
        self.command_execution_rate = 0  # Commands per minute for monitoring execution pace
        self.command_categories = {   # Track commands by category to ensure comprehensive testing
            "reconnaissance": set(),  # nmap, ping, etc
            "enumeration": set(),     # ls, find, cat, etc
            "web": set(),             # curl, wget, etc
            "exploitation": set(),    # exploit, nc, etc
            "privilege_escalation": set(),  # sudo, su, etc
            "brute_force": set(),     # hydra, john, etc
            "other": set()            # misc commands
        }
        self.min_category_coverage = 3  # Minimum number of categories that should be tried
        
        # Initialize component references
        self.memory = None
        self.rag_retriever = None
        self.prompt_generator = None
        self.task_tracker = None
        
        # Initialize available components
        if enable_memory and ADVANCED_MEMORY_AVAILABLE:
            self.init_memory_system(memory_params or {})
            
        if enable_rag and ENHANCED_RAG_AVAILABLE:
            self.init_rag_system(rag_params or {})
            
        if enable_dynamic_prompts and DYNAMIC_PROMPTING_AVAILABLE:
            self.init_prompt_system(default_reasoner_prompt, default_responder_prompt)
            
        if enable_task_tracking and TASK_TRACKING_AVAILABLE:
            self.init_task_system()
            
        # Connect components where possible
        self.connect_components()
        
        # Try to load persistent state
        self.load_state()
        
    def init_memory_system(self, params: Dict[str, Any]):
        """
        Initialize the advanced memory system.
        
        Args:
            params: Configuration parameters for memory
        """
        if not ADVANCED_MEMORY_AVAILABLE:
            return
            
        # Extract memory parameters with defaults
        short_term_limit = params.get("short_term_limit", 20)  # Reduced from 30
        long_term_limit = params.get("long_term_limit", 100)  # Reduced from 200
        working_memory_slots = params.get("working_memory_slots", 5)  # Reduced from 7
        memory_path = os.path.join(self.persistence_path, "memory")
        
        # Check for existing memory file
        memory_file = os.path.join(memory_path, f"memory_{self.session_id}.json")
        if os.path.exists(memory_file):
            try:
                # Try to load existing memory
                self.memory = AdvancedMemory.load(memory_file)
                self.components_initialized.add("memory")
                return
            except Exception as e:
                print(f"Warning: Failed to load existing memory: {e}")
        
        # Create new memory system
        self.memory = create_advanced_memory(
            session_id=self.session_id,
            persistence_path=memory_path
        )
        
        # Adjust memory settings if available
        if hasattr(self.memory, 'short_term_memory') and hasattr(self.memory.short_term_memory, 'maxlen'):
            # Create a new deque with the desired maxlen and copy over existing items
            existing_items = list(self.memory.short_term_memory)
            self.memory.short_term_memory = deque(existing_items, maxlen=short_term_limit)
        if hasattr(self.memory, 'long_term_limit'):
            self.memory.long_term_limit = long_term_limit
        if hasattr(self.memory, 'working_memory_slots'):
            self.memory.working_memory_slots = working_memory_slots
        
        self.components_initialized.add("memory")
        
    def init_rag_system(self, params: Dict[str, Any]):
        """
        Initialize the enhanced RAG system.
        
        Args:
            params: Configuration parameters for RAG
        """
        if not ENHANCED_RAG_AVAILABLE:
            return
            
        # Extract RAG parameters with defaults
        docs_path = params.get("docs_path", "docs")
        k = params.get("k", 6)
        chunk_size = params.get("chunk_size", 1024)
        chunk_overlap = params.get("chunk_overlap", 100)
        use_bm25 = params.get("use_bm25", True)
        hybrid_retrieval = params.get("hybrid_retrieval", True)
        
        # Create RAG retriever
        self.rag_retriever = load_static_rag_context(
            docs_path=docs_path,
            k=k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_bm25=use_bm25,
            hybrid_retrieval=hybrid_retrieval,
            memory=self.memory,  # Connect to memory if available
            use_llm_reranking=True
        )
        
        self.components_initialized.add("rag")
        
    def init_prompt_system(self, 
                          default_reasoner_prompt: Optional[str] = None,
                          default_responder_prompt: Optional[str] = None):
        """
        Initialize the dynamic prompt system.
        
        Args:
            default_reasoner_prompt: Default prompt for reasoner
            default_responder_prompt: Default prompt for responder
        """
        if not DYNAMIC_PROMPTING_AVAILABLE:
            return
            
        self.prompt_generator = create_prompt_generator(
            default_reasoner=default_reasoner_prompt,
            default_responder=default_responder_prompt,
            memory=self.memory  # Connect to memory if available
        )
        
        self.components_initialized.add("prompts")
        
    def init_task_system(self):
        """Initialize the task tracking system."""
        if not TASK_TRACKING_AVAILABLE:
            return
            
        task_path = os.path.join(self.persistence_path, "tasks")
        
        self.task_tracker = create_task_tracker(
            agent_id=self.session_id,
            persistence_path=task_path,
            memory=self.memory  # Connect to memory if available
        )
        
        self.components_initialized.add("tasks")
        
    def connect_components(self):
        """Connect components to enable data sharing between systems."""
        # Connect memory to other components if available
        if self.memory:
            if self.rag_retriever and hasattr(self.rag_retriever, 'set_memory'):
                self.rag_retriever.set_memory(self.memory)
                
            if self.prompt_generator and hasattr(self.prompt_generator, 'set_memory'):
                self.prompt_generator.set_memory(self.memory)
                
            if self.task_tracker and hasattr(self.task_tracker, 'integrate_with_memory'):
                self.task_tracker.integrate_with_memory(self.memory)
                
    def process_message(self, 
                       message: Union[str, Dict[str, Any]],
                       is_user_message: bool = True) -> Dict[str, Any]:
        """
        Process an incoming message through all context systems.
        
        Args:
            message: The message content (string or dict with role and content)
            is_user_message: Whether this is a user message (vs assistant)
            
        Returns:
            Dictionary with context information
        """
        # Quick return if there's nothing to process
        if not message:
            return {"message_processed": False, "timestamp": time.time()}
            
        # Convert string to message dict if needed
        if isinstance(message, str):
            message_dict = {
                "role": "user" if is_user_message else "assistant",
                "content": message
            }
        else:
            message_dict = message
            
        # Track the query if it's a user message
        if is_user_message:
            self.last_query = message_dict.get("content", "")
            
        # Add to memory system - most important component
        memory_reference = None
        if self.memory and is_user_message:  # Optimize to only process user messages for memory
            memory_reference = self.memory.add_message(message_dict)
            
            # Extract entities if it's a user message and content is substantial
            content = message_dict.get("content", "")
            if hasattr(self.memory, '_extract_entities') and len(content) > 10:
                self.memory._extract_entities(content)
                
        # Update prompt generator - only for user messages
        if self.prompt_generator and is_user_message:
            self.prompt_generator.update_from_query(message_dict.get("content", ""))
            
        # Update task tracking system - lower priority
        context_info = {}
        if self.task_tracker and is_user_message:  # Only process user messages for tasks
            # Update agent state
            self.task_tracker.agent_state.update_activity()
            
            # If we have an active task, add a note about the message
            current_task = self.task_tracker.get_current_task()
            if current_task:
                task_id = current_task.get("task_id")
                content = message_dict.get("content", "")
                if len(content) > 20:  # Only add substantial messages as notes
                    preview = content[:50] + ("..." if len(content) > 50 else "")
                    self.task_tracker.add_task_note(task_id, f"User: {preview}")
                
            # Add task info to context
            context_info["current_task"] = current_task
            context_info["next_tasks"] = self.task_tracker.get_next_tasks(limit=2)  # Reduced from 3
            
        # Return context information
        return {
            "memory_reference": memory_reference,
            "task_info": context_info,
            "message_processed": True,
            "timestamp": time.time()
        }
        
    def process_response(self, response: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process an outgoing response through all context systems.
        
        Args:
            response: The response content (string or dict with role and content)
            
        Returns:
            Dictionary with context information
        """
        # Process as a non-user message
        result = self.process_message(response, is_user_message=False)
        
        # Track the response
        if isinstance(response, str):
            self.last_response = response
        else:
            self.last_response = response.get("content", "")
            
        return result
        
    def get_enhanced_rag_documents(self, 
                                 query: str, 
                                 chat_history: Optional[List[Dict[str, Any]]] = None, 
                                 k: int = 3) -> Dict[str, Any]:
        """
        Get enhanced RAG documents for a query using all context systems.
        
        Args:
            query: The query string
            chat_history: Optional chat history
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with documents and context information
        """
        if not self.rag_retriever:
            return {"documents": [], "context_used": False}
            
        # Get proper context from memory if available
        memory_context = None
        if self.memory and hasattr(self.memory, 'get_context_for_query'):
            memory_context = self.memory.get_context_for_query(query)
            
        # Get documents with context
        docs = []
        try:
            if hasattr(self.rag_retriever, 'retrieve_with_context'):
                docs = self.rag_retriever.retrieve_with_context(
                    query=query,
                    chat_history=chat_history,
                    top_k=k
                )
            else:
                # Fallback to standard retrieval
                docs = self.rag_retriever.get_relevant_documents(query)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            
        return {
            "documents": docs,
            "memory_context": memory_context,
            "context_used": memory_context is not None,
            "document_count": len(docs)
        }
        
    def generate_optimized_prompt(self, 
                                is_reasoner: bool = True, 
                                chat_history: Optional[List[Dict[str, Any]]] = None,
                                query: Optional[str] = None) -> str:
        """
        Generate an optimized prompt based on context.
        
        Args:
            is_reasoner: Whether this is for the reasoner (vs responder)
            chat_history: Optional chat history
            query: Optional query string (uses last query if None)
            
        Returns:
            Optimized prompt text
        """
        if not self.prompt_generator:
            # Return empty prompt if generator not available
            return ""
            
        # Use last query if not provided
        if query is None:
            query = self.last_query
            
        # Generate optimized prompt
        if is_reasoner:
            return self.prompt_generator.generate_reasoner_prompt(chat_history, query)
        else:
            return self.prompt_generator.generate_responder_prompt(chat_history, query)
            
    def create_task(self, 
                  title: str, 
                  task_type: str,
                  description: Optional[str] = None,
                  priority: str = "medium") -> Optional[str]:
        """
        Create a new task in the task tracking system.
        
        Args:
            title: Task title
            task_type: Type of task
            description: Optional description
            priority: Task priority
            
        Returns:
            Task ID or None if task tracking is not available
        """
        if not self.task_tracker:
            return None
            
        return self.task_tracker.create_task(
            title=title,
            task_type=task_type,
            description=description,
            priority=priority
        )
        
    def start_task(self, task_id: str) -> bool:
        """
        Start a task and set it as the current task.
        
        Args:
            task_id: ID of the task to start
            
        Returns:
            True if task was started successfully
        """
        if not self.task_tracker:
            return False
            
        return self.task_tracker.start_task(task_id)
        
    def complete_task(self, task_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """
        Complete a task.
        
        Args:
            task_id: ID of the task to complete
            result: Optional task results
            
        Returns:
            True if task was completed successfully
        """
        if not self.task_tracker:
            return False
            
        return self.task_tracker.complete_task(task_id, result)
        
    def get_active_context(self) -> Dict[str, Any]:
        """
        Get the current active context from all systems.
        
        Returns:
            Dictionary with all context information
        """
        context = {
            "active_components": list(self.components_initialized),
            "session_id": self.session_id,
            "timestamp": time.time()
        }
        
        # Add memory context
        if self.memory:
            # Get working memory
            if hasattr(self.memory, 'working_memory'):
                context["working_memory"] = self.memory.working_memory
                
            # Get active entities
            if hasattr(self.memory, 'get_entities'):
                context["entities"] = self.memory.get_entities()[:5]  # Top 5 entities
                
            # Get recent conversations
            if hasattr(self.memory, 'conversations'):
                recent_convs = sorted(
                    self.memory.conversations, 
                    key=lambda c: c.end_time, 
                    reverse=True
                )[:2]  # Last 2 conversations
                context["recent_conversations"] = [c.summary for c in recent_convs]
                
        # Add task context
        if self.task_tracker:
            # Get current task
            current_task = self.task_tracker.get_current_task()
            if current_task:
                context["current_task"] = current_task
                
            # Get agent state summary
            context["agent_state"] = self.task_tracker.get_summary()
            
            # Get next tasks
            context["next_tasks"] = self.task_tracker.get_next_tasks(limit=2)
            
        # Add prompt context
        if self.prompt_generator:
            # Get conversation state
            if hasattr(self.prompt_generator, 'conversation_state'):
                context["conversation_state"] = self.prompt_generator.conversation_state
                
            # Get current entities
            if hasattr(self.prompt_generator, 'current_entities'):
                context["prompt_entities"] = self.prompt_generator.current_entities
                
        return context
        
    def save_state(self) -> Dict[str, bool]:
        """
        Save the state of all components.
        
        Returns:
            Dictionary with component save status
        """
        save_status = {}
        
        # Save memory state
        if self.memory and hasattr(self.memory, 'save'):
            try:
                memory_path = os.path.join(
                    self.persistence_path, 
                    f"memory_{self.session_id}.pickle"
                )
                self.memory.save(memory_path)
                save_status["memory"] = True
            except Exception as e:
                print(f"Error saving memory state: {e}")
                save_status["memory"] = False
                
        # Save task tracker state
        if self.task_tracker and hasattr(self.task_tracker, 'save_state'):
            try:
                task_path = self.task_tracker.save_state()
                save_status["tasks"] = task_path is not None
            except Exception as e:
                print(f"Error saving task state: {e}")
                save_status["tasks"] = False
                
        # Save overall context state
        try:
            context_state = {
                "session_id": self.session_id,
                "created_at": self.created_at,
                "last_saved": time.time(),
                "components_initialized": list(self.components_initialized),
                "last_query": self.last_query,
                "last_response": self.last_response,
                "conversation_active": self.conversation_active,
                "executed_commands": list(self.executed_commands),
                "command_results": self.command_results
            }
            
            context_path = os.path.join(
                self.persistence_path,
                f"context_state_{self.session_id}.json"
            )
            
            with open(context_path, 'w') as f:
                json.dump(context_state, f, indent=2)
                
            save_status["context"] = True
        except Exception as e:
            print(f"Error saving context state: {e}")
            save_status["context"] = False
            
        return save_status
        
    def load_state(self) -> Dict[str, bool]:
        """
        Load the state of all components.
        
        Returns:
            Dictionary with component load status
        """
        load_status = {}
        
        # Try to load context state first
        context_path = os.path.join(
            self.persistence_path,
            f"context_state_{self.session_id}.json"
        )
        
        try:
            if os.path.exists(context_path):
                with open(context_path, 'r') as f:
                    context_state = json.load(f)
                    
                self.created_at = context_state.get("created_at", self.created_at)
                self.last_query = context_state.get("last_query")
                self.last_response = context_state.get("last_response")
                self.conversation_active = context_state.get("conversation_active", True)
                
                # Restore command tracking
                if "executed_commands" in context_state:
                    self.executed_commands = set(context_state.get("executed_commands", []))
                if "command_results" in context_state:
                    self.command_results = context_state.get("command_results", {})
                
                load_status["context"] = True
            else:
                load_status["context"] = False
        except Exception as e:
            print(f"Error loading context state: {e}")
            load_status["context"] = False
            
        # Load memory state
        if self.memory and hasattr(self.memory, 'load'):
            try:
                memory_path = os.path.join(
                    self.persistence_path, 
                    f"memory_{self.session_id}.pickle"
                )
                
                if os.path.exists(memory_path):
                    self.memory = AdvancedMemory.load(memory_path)
                    load_status["memory"] = True
                else:
                    load_status["memory"] = False
            except Exception as e:
                print(f"Error loading memory state: {e}")
                load_status["memory"] = False
                
        # Load task tracker state
        if self.task_tracker and hasattr(self.task_tracker, 'load_state'):
            try:
                loaded = self.task_tracker.load_state()
                load_status["tasks"] = loaded
            except Exception as e:
                print(f"Error loading task state: {e}")
                load_status["tasks"] = False
                
        # Reconnect components after loading
        self.connect_components()
        
        return load_status
    
    def extract_insights_from_conversation(self) -> Dict[str, Any]:
        """
        Extract insights from the conversation using all context systems.
        
        Returns:
            Dictionary with extracted insights
        """
        insights = {
            "entities": [],
            "tasks": [],
            "conversation_topics": [],
            "key_findings": []
        }
        
        # Extract entities from memory
        if self.memory and hasattr(self.memory, 'get_entities'):
            insights["entities"] = self.memory.get_entities()
            
        # Extract tasks from task tracker
        if self.task_tracker:
            insights["tasks"] = self.task_tracker.get_session_tasks()
            
        # Extract conversation topics
        if self.memory and hasattr(self.memory, 'conversations'):
            topics = set()
            for conv in self.memory.conversations:
                topics.update(conv.topics)
            insights["conversation_topics"] = list(topics)
            
        return insights


# Helper function to create a context manager
def create_context_manager(session_id: Optional[str] = None, 
                         persistence_path: Optional[str] = "./context_data",
                         default_reasoner_prompt: Optional[str] = None,
                         default_responder_prompt: Optional[str] = None) -> ContextManager:
    """
    Create and initialize a context manager.
    
    Args:
        session_id: Optional unique session identifier
        persistence_path: Path for persisting context data
        default_reasoner_prompt: Default prompt for reasoner
        default_responder_prompt: Default prompt for responder
        
    Returns:
        Initialized ContextManager
    """
    return ContextManager(
        session_id=session_id,
        persistence_path=persistence_path,
        enable_memory=ADVANCED_MEMORY_AVAILABLE,
        enable_rag=ENHANCED_RAG_AVAILABLE,
        enable_dynamic_prompts=DYNAMIC_PROMPTING_AVAILABLE,
        enable_task_tracking=TASK_TRACKING_AVAILABLE,
        default_reasoner_prompt=default_reasoner_prompt,
        default_responder_prompt=default_responder_prompt
    )


# Integration with DarkCircuit agent
def integrate_with_darkcircuit_agent(agent, context_manager: ContextManager):
    """
    Integrate a context manager with a DarkCircuit agent.
    
    Args:
        agent: The DarkCircuit agent instance
        context_manager: The context manager to integrate
    """
    # Connect memory system
    if hasattr(agent, 'chat_history') and context_manager.memory:
        # Initialize memory with existing chat history
        for msg in agent.chat_history:
            if isinstance(msg, dict):
                context_manager.process_message(msg)
            else:
                # Convert to dict based on type
                role = getattr(msg, 'type', None) or getattr(msg, 'role', 'unknown')
                content = getattr(msg, 'content', str(msg))
                context_manager.process_message({
                    "role": role,
                    "content": content
                })
    
    # Override the agent's prompts with dynamic ones
    if context_manager.prompt_generator:
        def get_dynamic_reasoner_prompt():
            return context_manager.generate_optimized_prompt(is_reasoner=True)
            
        def get_dynamic_responder_prompt():
            return context_manager.generate_optimized_prompt(is_reasoner=False)
            
        # Attach the dynamic prompt getters to the agent
        agent.get_reasoner_prompt = get_dynamic_reasoner_prompt
        agent.get_responder_prompt = get_dynamic_responder_prompt
    
    # Enhance the RAG capabilities
    if context_manager.rag_retriever and hasattr(agent, 'rag_retrieve_with_context'):
        original_rag = agent.rag_retrieve_with_context
        
        def enhanced_rag_retrieve(query, chat_history=None):
            # Get enhanced RAG results
            rag_results = context_manager.get_enhanced_rag_documents(
                query=query,
                chat_history=chat_history
            )
            
            # If we got results, format and return them
            if rag_results["documents"]:
                content_parts = []
                for i, doc in enumerate(rag_results["documents"]):
                    metadata = doc.metadata
                    source = metadata.get("source", "Unknown source")
                    content_parts.append(f"[Source {i + 1}: {source}] {doc.page_content}")
                
                return "\n\n".join(content_parts)
            
            # Fall back to original implementation
            return original_rag(query, chat_history)
            
        # Replace the agent's RAG function
        agent.rag_retrieve_with_context = enhanced_rag_retrieve
    
    # Add task tracking support
    if context_manager.task_tracker:
        # Add task creation function to agent
        agent.create_task = context_manager.create_task
        agent.start_task = context_manager.start_task
        agent.complete_task = context_manager.complete_task
        agent.get_active_context = context_manager.get_active_context
    
    # Add command tracking
    original_run_command = getattr(agent, 'run_command', None)
    
    if original_run_command:
        async def enhanced_run_command(command: str) -> str:
            # Only prevent exact duplicate commands, allow variations
            exact_match = command in context_manager.executed_commands
            
            # But allow common commands to be repeated occasionally based on context
            allow_repeat = False
            
            # Always allow these basic commands to repeat
            basic_repeatable_commands = ["ls", "ls -la", "pwd", "whoami", "id", "cat /etc/passwd", "ps aux"]
            if any(command.strip() == basic for basic in basic_repeatable_commands):
                allow_repeat = True
            
            # Check if we have too few total commands - if so, permit some repetition 
            # to ensure continued execution
            if len(context_manager.executed_commands) < 15:
                allow_repeat = True
            
            # Check if we've already run this command but don't block progression
            if exact_match and not allow_repeat:
                print(f"[Context] Detected repeated command: {command}")
                if command in context_manager.command_results:
                    print(f"[Context] Returning cached result")
                    # Handle different result formats
                    if isinstance(context_manager.command_results[command], dict) and "result" in context_manager.command_results[command]:
                        cached_result = context_manager.command_results[command]["result"]
                    else:
                        cached_result = context_manager.command_results[command]
                        
                    return f"[CACHED RESULT] This command was already executed. Previous result:\n\n{cached_result}"
            
            # Execute the command
            result = await original_run_command(command)
            
            # Add timestamp for monitoring execution rate
            timestamp = time.time()
            context_manager.command_timestamps.append(timestamp)
            
            # Calculate command execution rate (commands per minute)
            if len(context_manager.command_timestamps) > 1:
                # Calculate rate based on last 10 commands or all commands if fewer
                recent_timestamps = context_manager.command_timestamps[-10:]
                if len(recent_timestamps) > 1:
                    time_span = recent_timestamps[-1] - recent_timestamps[0]  # seconds
                    if time_span > 0:
                        commands_count = len(recent_timestamps) - 1
                        context_manager.command_execution_rate = (commands_count / time_span) * 60  # convert to per minute
            
            # Categorize the command to track coverage
            categorized = False
            if any(cmd in command.lower() for cmd in ["nmap", "ping", "traceroute", "host", "dig", "whois"]):
                context_manager.command_categories["reconnaissance"].add(command)
                categorized = True
            if any(cmd in command.lower() for cmd in ["ls", "find", "locate", "cat", "grep", "dir", "type"]):
                context_manager.command_categories["enumeration"].add(command)
                categorized = True
            if any(cmd in command.lower() for cmd in ["curl", "wget", "http", "dirb", "gobuster", "nikto", "browser"]):
                context_manager.command_categories["web"].add(command)
                categorized = True
            if any(cmd in command.lower() for cmd in ["exploit", "nc", "netcat", "rev", "shell", "msfconsole", "msf"]):
                context_manager.command_categories["exploitation"].add(command)
                categorized = True
            if any(cmd in command.lower() for cmd in ["sudo", "su", "chmod", "chown", "setuid", "suid", "root"]):
                context_manager.command_categories["privilege_escalation"].add(command)
                categorized = True
            if any(cmd in command.lower() for cmd in ["hydra", "john", "hashcat", "crack", "brute", "password"]):
                context_manager.command_categories["brute_force"].add(command)
                categorized = True
            if not categorized:
                context_manager.command_categories["other"].add(command)
            
            # Track the command and its result
            context_manager.executed_commands.add(command)
            
            # Store result with timestamp for chronological reference
            if isinstance(context_manager.command_results, dict):
                # Handle possible dict-of-dicts case
                try:
                    context_manager.command_results[command] = {
                        "result": result,
                        "timestamp": timestamp,
                        "execution_count": len(context_manager.executed_commands)
                    }
                except Exception as e:
                    print(f"[Context] Warning: Error storing command result in structured format: {e}")
                    # Fallback to simple string storage
                    context_manager.command_results[command] = result
            else:
                # Fallback for simple string dictionary
                context_manager.command_results[command] = result
            
            # Append completion status to result for more context
            categories_covered = sum(1 for category, commands in context_manager.command_categories.items() if commands)
            
            # Add execution statistics if we've run several commands
            if len(context_manager.executed_commands) > 5:
                completion_status = f"\n\n[Execution Stats] Commands: {len(context_manager.executed_commands)}, Categories: {categories_covered}/{len(context_manager.command_categories)}, Rate: {context_manager.command_execution_rate:.1f} cmd/min"
                # Only add stats if this is a complex command result (avoid cluttering simple outputs)
                if len(result.split("\n")) > 5:
                    result += completion_status
            
            return result
            
        # Replace the agent's command function
        agent.run_command = enhanced_run_command
    
    # Add hook for processing messages
    original_run_agent = getattr(agent, 'run_agent_streaming', None)
    
    if original_run_agent:
        async def enhanced_run_agent(user_prompt):
            # Process user message through context manager
            try:
                context_manager.process_message(user_prompt)
            except Exception as e:
                print(f"Warning: Error processing message in context manager: {e}")
            
            try:
                # Run original agent with error handling
                async for event in original_run_agent(user_prompt):
                    yield event
            except Exception as e:
                print(f"Error in agent execution: {e}")
                # Return a graceful error message
                yield {
                    "type": "token", 
                    "value": "\nI'm sorry, I encountered a processing error. Let's try a simpler approach. Please ask your question again."
                }
                
            # Process agent's response
            try:
                if hasattr(agent, 'chat_history') and len(agent.chat_history) > 0:
                    last_msg = agent.chat_history[-1]
                    if hasattr(last_msg, 'type') and last_msg.type == 'ai':
                        context_manager.process_response(last_msg.content)
                    elif hasattr(last_msg, 'role') and last_msg.role == 'assistant':
                        context_manager.process_response(last_msg.content)
            except Exception as e:
                print(f"Warning: Error processing response in context manager: {e}")
                    
            # Save state after each interaction
            try:
                context_manager.save_state()
            except Exception as e:
                print(f"Warning: Error during context state saving: {e}")
                # Continue execution even if saving fails
            
        # Replace the agent's run function
        agent.run_agent_streaming = enhanced_run_agent
        
        # Store reference to context manager in agent for cleanup
        agent.context_manager = context_manager


# Initialize context manager with DarkCircuit agent
def init_darkcircuit_context(agent, 
                           session_id: Optional[str] = None,
                           persistence_path: Optional[str] = "./context_data"):
    """
    Initialize and integrate a context manager with a DarkCircuit agent.
    
    Args:
        agent: The DarkCircuit agent instance
        session_id: Optional unique session identifier
        persistence_path: Path for persisting context data
        
    Returns:
        The created and integrated context manager
    """
    # Get default prompts from agent
    default_reasoner_prompt = getattr(agent, 'reasoning_prompt', None)
    if default_reasoner_prompt and hasattr(default_reasoner_prompt, 'content'):
        default_reasoner_prompt = default_reasoner_prompt.content
        
    default_responder_prompt = getattr(agent, 'response_prompt', None)
    if default_responder_prompt and hasattr(default_responder_prompt, 'content'):
        default_responder_prompt = default_responder_prompt.content
    
    # Create context manager
    context_manager = create_context_manager(
        session_id=session_id,
        persistence_path=persistence_path,
        default_reasoner_prompt=default_reasoner_prompt,
        default_responder_prompt=default_responder_prompt
    )
    
    # Integrate with agent
    integrate_with_darkcircuit_agent(agent, context_manager)
    
    return context_manager