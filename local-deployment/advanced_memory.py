"""
Advanced Memory Management for DarkCircuit Agent

This module provides sophisticated memory management capabilities,
including hierarchical memory structures, conversation summarization,
entity tracking, and relevance scoring.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import deque
import pickle
from datetime import datetime
import hashlib

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

class MemoryItem:
    """Base class for memory items that can be stored in the memory system."""
    
    def __init__(self, content: Any, timestamp: Optional[float] = None, importance: float = 0.5):
        self.content = content
        self.timestamp = timestamp or time.time()
        self.importance = importance
        self.creation_date = datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        self.access_count = 0
        self.last_accessed = self.timestamp
        self.decay_rate = 0.95  # How quickly importance decays over time
        
    def access(self):
        """Record an access to this memory item."""
        self.access_count += 1
        self.last_accessed = time.time()
        
    def calculate_relevance(self, query: str, current_time: Optional[float] = None) -> float:
        """Calculate relevance score based on importance, recency, and query similarity."""
        current_time = current_time or time.time()
        time_factor = self.decay_rate ** ((current_time - self.timestamp) / (60 * 60))  # Decay per hour
        recency_factor = self.decay_rate ** ((current_time - self.last_accessed) / (60 * 60))
        frequency_factor = min(1.0, self.access_count / 10)  # Cap at 1.0
        
        # Simple text similarity (this could be enhanced with embeddings)
        if isinstance(self.content, str) and isinstance(query, str):
            content_lower = self.content.lower()
            query_lower = query.lower()
            overlap = sum(1 for word in query_lower.split() if word in content_lower)
            similarity = overlap / max(1, len(query_lower.split()))
        else:
            similarity = 0.0
            
        # Combined relevance score
        relevance = (
            0.3 * self.importance +
            0.2 * time_factor +
            0.2 * recency_factor +
            0.1 * frequency_factor +
            0.2 * similarity
        )
        
        return min(1.0, max(0.0, relevance))  # Ensure score is between 0 and 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation for serialization."""
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "creation_date": self.creation_date,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create a memory item from a dictionary representation."""
        item = cls(content=data["content"], timestamp=data["timestamp"], importance=data["importance"])
        item.creation_date = data["creation_date"]
        item.access_count = data["access_count"]
        item.last_accessed = data["last_accessed"]
        return item


class MessageMemoryItem(MemoryItem):
    """Memory item specifically for conversation messages."""
    
    def __init__(self, message: Dict[str, Any], importance: float = 0.5):
        super().__init__(content=message, timestamp=time.time(), importance=importance)
        self.role = message.get("role", "") or message.get("type", "unknown")
        self.text = message.get("content", "")
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict["role"] = self.role
        base_dict["text"] = self.text
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessageMemoryItem':
        """Create a message memory item from a dictionary."""
        item = super().from_dict(data)
        item.role = data["role"]
        item.text = data["text"]
        return item


class Entity:
    """Represents an entity detected in conversation (IPs, tools, vulnerabilities, etc.)."""
    
    def __init__(self, name: str, entity_type: str, value: Any, first_mentioned: float = None):
        self.name = name
        self.entity_type = entity_type
        self.value = value
        self.first_mentioned = first_mentioned or time.time()
        self.last_mentioned = self.first_mentioned
        self.mention_count = 1
        self.importance = 0.5
        self.relationships: Dict[str, List[str]] = {}  # Relationships to other entities
        
    def mention(self):
        """Record a new mention of this entity."""
        self.last_mentioned = time.time()
        self.mention_count += 1
        
    def add_relationship(self, relationship_type: str, related_entity: str):
        """Add a relationship to another entity."""
        if relationship_type not in self.relationships:
            self.relationships[relationship_type] = []
        if related_entity not in self.relationships[relationship_type]:
            self.relationships[relationship_type].append(related_entity)
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "value": self.value,
            "first_mentioned": self.first_mentioned,
            "last_mentioned": self.last_mentioned,
            "mention_count": self.mention_count,
            "importance": self.importance,
            "relationships": self.relationships
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create an entity from a dictionary."""
        entity = cls(
            name=data["name"],
            entity_type=data["entity_type"],
            value=data["value"],
            first_mentioned=data["first_mentioned"]
        )
        entity.last_mentioned = data["last_mentioned"]
        entity.mention_count = data["mention_count"]
        entity.importance = data["importance"]
        entity.relationships = data["relationships"]
        return entity


class Conversation:
    """Represents a conversation summary."""
    
    def __init__(self, summary: str, start_time: float, end_time: float, 
                 topics: List[str], message_ids: List[str], importance: float = 0.5):
        self.summary = summary
        self.start_time = start_time
        self.end_time = end_time
        self.topics = topics
        self.message_ids = message_ids
        self.importance = importance
        self.last_accessed = time.time()
        self.access_count = 0
        
    def access(self):
        """Record an access to this conversation summary."""
        self.access_count += 1
        self.last_accessed = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "summary": self.summary,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "topics": self.topics,
            "message_ids": self.message_ids,
            "importance": self.importance,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create a conversation from a dictionary."""
        conversation = cls(
            summary=data["summary"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            topics=data["topics"],
            message_ids=data["message_ids"],
            importance=data["importance"]
        )
        conversation.last_accessed = data["last_accessed"]
        conversation.access_count = data["access_count"]
        return conversation


class AdvancedMemory:
    """
    Sophisticated memory system with short-term, long-term, and working memory components.
    Includes entity tracking, conversation summarization, and relevance scoring.
    """
    
    def __init__(self, 
                 short_term_limit: int = 20, 
                 long_term_limit: int = 100,
                 working_memory_slots: int = 7,
                 session_id: Optional[str] = None,
                 persistence_path: Optional[str] = None,
                 summarization_model: str = "gpt-3.5-turbo"):
        
        self.session_id = session_id or hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.creation_time = time.time()
        
        # Memory components
        self.short_term_memory: deque = deque(maxlen=short_term_limit)
        self.long_term_memory: List[MessageMemoryItem] = []
        self.long_term_limit = long_term_limit
        self.working_memory: List[Any] = []
        self.working_memory_slots = working_memory_slots
        
        # Entity tracking
        self.entities: Dict[str, Entity] = {}
        
        # Conversation summarization
        self.conversations: List[Conversation] = []
        self.pending_summary_messages: List[MessageMemoryItem] = []
        self.summary_threshold = 15  # Increased from 10 to reduce summarization frequency
        
        # Persistence
        self.persistence_path = persistence_path
        if persistence_path and not os.path.exists(persistence_path):
            os.makedirs(persistence_path, exist_ok=True)
        
        # Summarization
        try:
            self.summarization_model = ChatOpenAI(model=summarization_model)
        except Exception as e:
            print(f"Warning: Failed to initialize summarization model: {e}")
            self.summarization_model = None
    
    def add_message(self, message: Dict[str, Any], importance: Optional[float] = None):
        """
        Add a message to memory and process it for entities and context.
        
        Args:
            message: Message dict with role/type and content
            importance: Optional importance override (0.0 to 1.0)
        """
        # Auto-evaluate importance if not specified
        if importance is None:
            importance = self._evaluate_importance(message)
            
        # Create memory item
        memory_item = MessageMemoryItem(message, importance=importance)
        
        # Add to short-term memory
        self.short_term_memory.append(memory_item)
        
        # Add to pending summary queue
        self.pending_summary_messages.append(memory_item)
        
        # Extract and track entities
        if message.get("role") in ["user", "human"] or message.get("type") in ["user", "human"]:
            self._extract_entities(message.get("content", ""))
        
        # Create summary if we've reached the threshold
        if len(self.pending_summary_messages) >= self.summary_threshold:
            self._create_conversation_summary()
        
        # Update working memory
        self._update_working_memory()
        
        # Return unique ID for this message
        return f"msg_{int(memory_item.timestamp * 1000)}_{hash(str(message)) % 10000}"
    
    def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memory for relevant items using the query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of memory items sorted by relevance
        """
        # Collect memory items with relevance scores
        scored_items = []
        
        # Check short-term memory
        for item in self.short_term_memory:
            relevance = item.calculate_relevance(query)
            scored_items.append((item, relevance, "short_term"))
            
        # Check long-term memory
        for item in self.long_term_memory:
            relevance = item.calculate_relevance(query)
            scored_items.append((item, relevance, "long_term"))
            
        # Check conversation summaries
        for conv in self.conversations:
            # Calculate relevance based on topics and summary
            query_terms = set(query.lower().split())
            topic_overlap = len([t for t in conv.topics if any(term in t.lower() for term in query_terms)])
            summary_relevance = conv.calculate_relevance(query) if hasattr(conv, 'calculate_relevance') else 0.5
            
            # Combined relevance score
            relevance = 0.7 * summary_relevance + 0.3 * (topic_overlap / max(1, len(query_terms)))
            scored_items.append((conv, relevance, "conversation"))
        
        # Sort by relevance score (descending)
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        results = []
        for item, score, source in scored_items[:limit]:
            if hasattr(item, 'to_dict'):
                item_dict = item.to_dict()
            else:
                item_dict = {"content": str(item)}
                
            item_dict["relevance"] = score
            item_dict["source"] = source
            results.append(item_dict)
            
            # Record access
            if hasattr(item, 'access'):
                item.access()
                
        return results
    
    def get_entities(self, entity_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get tracked entities, optionally filtered by type.
        
        Args:
            entity_types: Optional list of entity types to filter by
            
        Returns:
            List of entity dictionaries
        """
        if entity_types:
            return [e.to_dict() for e in self.entities.values() 
                   if e.entity_type in entity_types]
        return [e.to_dict() for e in self.entities.values()]
    
    def get_context_for_query(self, query: str, token_limit: int = 1500) -> str:
        """
        Generate a context string for a query by combining:
        1. Working memory
        2. Relevant entities
        3. Recent messages
        4. Related conversation summaries
        
        Args:
            query: The query to generate context for
            token_limit: Approximate token limit for context
            
        Returns:
            Formatted context string
        """
        context_parts = []
        estimated_tokens = 0
        
        # Add working memory
        if self.working_memory:
            working_mem_str = "Working Memory:\n" + "\n".join(
                [f"- {item}" for item in self.working_memory]
            )
            context_parts.append(working_mem_str)
            estimated_tokens += len(working_mem_str.split()) * 1.3  # Rough token estimation
        
        # Add relevant entities
        # First, score entities by relevance to query
        entity_scores = []
        for entity in self.entities.values():
            # Simple relevance check based on name/type match
            relevance = 0.0
            if entity.name.lower() in query.lower():
                relevance += 0.7
            if entity.entity_type.lower() in query.lower():
                relevance += 0.3
            if relevance > 0:
                entity_scores.append((entity, relevance))
        
        # Sort by score and add top entities
        if entity_scores:
            entity_scores.sort(key=lambda x: x[1], reverse=True)
            entities_str = "Relevant Entities:\n" + "\n".join(
                [f"- {e.name} ({e.entity_type}): {e.value}" for e, _ in entity_scores[:3]]
            )
            context_parts.append(entities_str)
            estimated_tokens += len(entities_str.split()) * 1.3
        
        # Add recent conversation summaries
        relevant_summaries = self.search_memory(query, limit=2)
        summaries = [item for item in relevant_summaries if item["source"] == "conversation"]
        if summaries:
            summaries_str = "Conversation Context:\n" + "\n".join(
                [f"- {s['content']['summary']}" for s in summaries]
            )
            context_parts.append(summaries_str)
            estimated_tokens += len(summaries_str.split()) * 1.3
        
        # Add recent messages if we have room
        if estimated_tokens < token_limit * 0.7:
            recent_msgs = list(self.short_term_memory)[-5:]  # Last 5 messages
            if recent_msgs:
                messages_str = "Recent Messages:\n" + "\n".join(
                    [f"- {msg.role}: {msg.text}" for msg in recent_msgs]
                )
                context_parts.append(messages_str)
        
        return "\n\n".join(context_parts)
    
    def save(self, file_path: Optional[str] = None) -> str:
        """
        Save the memory state to disk.
        
        Args:
            file_path: Optional explicit file path, otherwise uses persistence_path
            
        Returns:
            Path where memory was saved
        """
        if file_path is None:
            if self.persistence_path is None:
                raise ValueError("No persistence path specified")
            file_path = os.path.join(self.persistence_path, f"memory_{self.session_id}.json")
        
        # Create a serializable version of the memory state
        memory_state = {
            "session_id": self.session_id,
            "creation_time": self.creation_time,
            "short_term_memory": [msg.to_dict() for msg in self.short_term_memory],
            "long_term_memory": [msg.to_dict() for msg in self.long_term_memory],
            "working_memory": self.working_memory.copy() if hasattr(self, 'working_memory') else [],
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "conversations": [c.to_dict() for c in self.conversations],
            "pending_summary_messages": [msg.to_dict() for msg in self.pending_summary_messages],
            "summary_threshold": self.summary_threshold
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(memory_state, f, indent=2)
            return file_path
        except Exception as e:
            print(f"Warning: Failed to save memory state: {e}")
            # Try a more minimal version if full serialization fails
            try:
                minimal_state = {
                    "session_id": self.session_id,
                    "creation_time": self.creation_time,
                    "short_term_messages": [
                        {"role": msg.role, "text": msg.text} for msg in self.short_term_memory
                    ],
                    "entities": [
                        {"name": e.name, "entity_type": e.entity_type, "value": e.value}
                        for e in self.entities.values()
                    ]
                }
                backup_path = file_path.replace('.json', '_minimal.json')
                with open(backup_path, 'w') as f:
                    json.dump(minimal_state, f)
                return backup_path
            except Exception as inner_e:
                print(f"Warning: Failed to save minimal memory state: {inner_e}")
                return ""
    
    @classmethod
    def load(cls, file_path: str) -> 'AdvancedMemory':
        """
        Load a memory state from disk.
        
        Args:
            file_path: Path to the memory state file
            
        Returns:
            Loaded AdvancedMemory instance
        """
        try:
            with open(file_path, 'r') as f:
                memory_state = json.load(f)
            
            # Create a new memory instance
            memory = cls(
                session_id=memory_state.get("session_id"),
                persistence_path=os.path.dirname(file_path)
            )
            
            # Restore short-term memory
            if "short_term_memory" in memory_state:
                for msg_data in memory_state["short_term_memory"]:
                    if isinstance(msg_data, dict):
                        msg = MessageMemoryItem(msg_data)
                        memory.short_term_memory.append(msg)
            
            # Restore entities
            if "entities" in memory_state and isinstance(memory_state["entities"], dict):
                for entity_id, entity_data in memory_state["entities"].items():
                    if isinstance(entity_data, dict):
                        entity = Entity(
                            name=entity_data.get("name", ""),
                            entity_type=entity_data.get("entity_type", "unknown"),
                            value=entity_data.get("value", "")
                        )
                        memory.entities[entity_id] = entity
            
            # Restore conversations if present
            if "conversations" in memory_state:
                for conv_data in memory_state["conversations"]:
                    if isinstance(conv_data, dict):
                        try:
                            conv = Conversation(
                                summary=conv_data.get("summary", ""),
                                start_time=conv_data.get("start_time", 0),
                                end_time=conv_data.get("end_time", 0),
                                topics=conv_data.get("topics", []),
                                message_ids=conv_data.get("message_ids", [])
                            )
                            memory.conversations.append(conv)
                        except Exception as e:
                            print(f"Error restoring conversation: {e}")
            
            return memory
            
        except Exception as e:
            print(f"Error loading memory state: {e}")
            # Return a new empty memory instance
            return cls(
                session_id=os.path.basename(file_path).replace("memory_", "").replace(".json", ""),
                persistence_path=os.path.dirname(file_path)
            )
    
    def _evaluate_importance(self, message: Dict[str, Any]) -> float:
        """
        Evaluate the importance of a message based on heuristics.
        
        Args:
            message: The message to evaluate
            
        Returns:
            Importance score (0.0 to 1.0)
        """
        content = message.get("content", "")
        if not content:
            return 0.3  # Empty messages are less important
        
        importance = 0.5  # Default importance
        
        # Check for entities
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        url_pattern = r'https?://[^\s]+'
        code_pattern = r'```[\s\S]*?```'
        
        if any(p in content.lower() for p in ['ip address', 'target', 'machine', 'htb', 'hack the box']):
            importance += 0.2
        
        # Messages with specific entities are more important
        if any(p in content for p in ['CVE-', 'exploit', 'vulnerability']):
            importance += 0.15
            
        # Messages with commands or actions are important
        if any(p in content.lower() for p in ['run', 'execute', 'command', 'scan']):
            importance += 0.1
            
        # Question messages might be important
        if '?' in content:
            importance += 0.05
            
        # Longer, more detailed messages might be more important
        if len(content) > 200:
            importance += 0.05
            
        return min(1.0, importance)  # Cap at 1.0
    
    def _extract_entities(self, text: str):
        """
        Extract entities from text and add them to tracking.
        
        Args:
            text: Text to extract entities from
        """
        # Simple IP address extraction
        import re
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        ips = re.findall(ip_pattern, text)
        
        for ip in ips:
            entity_id = f"ip_{ip}"
            if entity_id in self.entities:
                self.entities[entity_id].mention()
            else:
                self.entities[entity_id] = Entity(
                    name=ip,
                    entity_type="ip_address",
                    value=ip
                )
                
        # Extract potential tool names
        tool_patterns = [
            (r'\bnmap\b', "scanner"),
            (r'\bgobuster\b', "directory_scanner"),
            (r'\bdirbuster\b', "directory_scanner"),
            (r'\bhydra\b', "password_cracker"),
            (r'\bmetasploit\b', "exploitation_framework"),
            (r'\bmsfvenom\b', "payload_generator"),
            (r'\bcurl\b', "http_client"),
            (r'\bwget\b', "http_client"),
            (r'\bssh\b', "remote_access"),
            (r'\bexploit\b', "exploit"),
            (r'\bCVE-\d{4}-\d{4,7}\b', "vulnerability")
        ]
        
        for pattern, entity_type in tool_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_id = f"{entity_type}_{match.lower()}"
                if entity_id in self.entities:
                    self.entities[entity_id].mention()
                else:
                    self.entities[entity_id] = Entity(
                        name=match,
                        entity_type=entity_type,
                        value=match
                    )
                    
        # Try to link related entities
        self._link_entities()
    
    def _link_entities(self):
        """Link related entities based on temporal proximity and type relationships."""
        # Get recently mentioned entities (last 5 minutes)
        current_time = time.time()
        recent_entities = [
            e for e in self.entities.values()
            if current_time - e.last_mentioned < 300  # 5 minutes
        ]
        
        if len(recent_entities) < 2:
            return
            
        # Build links based on entity types
        for i, entity1 in enumerate(recent_entities):
            for entity2 in recent_entities[i+1:]:
                # Link IPs with tools used against them
                if entity1.entity_type == "ip_address" and entity2.entity_type in [
                    "scanner", "directory_scanner", "http_client", "exploit"
                ]:
                    entity1.add_relationship("scanned_by", entity2.name)
                    entity2.add_relationship("scanned", entity1.name)
                elif entity2.entity_type == "ip_address" and entity1.entity_type in [
                    "scanner", "directory_scanner", "http_client", "exploit"
                ]:
                    entity2.add_relationship("scanned_by", entity1.name)
                    entity1.add_relationship("scanned", entity2.name)
                    
                # Link vulnerabilities with exploitation tools
                if entity1.entity_type == "vulnerability" and entity2.entity_type in [
                    "exploit", "exploitation_framework"
                ]:
                    entity1.add_relationship("exploited_by", entity2.name)
                    entity2.add_relationship("exploits", entity1.name)
                elif entity2.entity_type == "vulnerability" and entity1.entity_type in [
                    "exploit", "exploitation_framework"
                ]:
                    entity2.add_relationship("exploited_by", entity1.name)
                    entity1.add_relationship("exploits", entity2.name)
    
    def _update_working_memory(self):
        """Update the working memory with the most important current context."""
        # Clear current working memory
        self.working_memory = []
        
        # Add active entities by importance (using mention count as a proxy)
        entities_by_importance = sorted(
            self.entities.values(),
            key=lambda e: e.mention_count * (0.7 + 0.3 * (1.0 - min(1.0, (time.time() - e.last_mentioned) / 3600))),
            reverse=True
        )
        
        # Add top entities to working memory
        for entity in entities_by_importance[:3]:
            # Format entity with its relationships for better context
            entity_str = f"{entity.name} ({entity.entity_type})"
            for rel_type, related in entity.relationships.items():
                if related:
                    entity_str += f" {rel_type}: {', '.join(related[:3])}"
            
            self.working_memory.append(entity_str)
            
        # Add most recent conversation topic if we have one
        if self.conversations:
            latest_conv = max(self.conversations, key=lambda c: c.end_time)
            if time.time() - latest_conv.end_time < 1800:  # 30 minutes
                self.working_memory.append(f"Recent conversation: {latest_conv.summary}")
                
        # Fill remaining slots with important messages from short-term memory
        important_messages = sorted(
            self.short_term_memory,
            key=lambda m: m.importance,
            reverse=True
        )
        
        slots_remaining = self.working_memory_slots - len(self.working_memory)
        for msg in important_messages[:slots_remaining]:
            # Format as a concise reference
            role = msg.role
            content_preview = msg.text[:50] + "..." if len(msg.text) > 50 else msg.text
            self.working_memory.append(f"{role}: {content_preview}")
    
    def _create_conversation_summary(self):
        """Create a summary of the pending messages and add to conversations."""
        if not self.pending_summary_messages or not self.summarization_model:
            self.pending_summary_messages = []
            return
            
        try:
            # Get messages to summarize
            messages_to_summarize = self.pending_summary_messages
            self.pending_summary_messages = []
            
            if not messages_to_summarize:
                return
                
            # Format messages for the summarization model
            formatted_messages = []
            for msg in messages_to_summarize:
                role = "user" if msg.role in ["user", "human"] else "assistant"
                formatted_messages.append({"role": role, "content": msg.text})
                
            # Create system message with instructions
            system_message = {
                "role": "system",
                "content": (
                    "Summarize the following conversation segment concisely. "
                    "Identify key topics, entities mentioned, and important information. "
                    "Focus on security-relevant details like IP addresses, attack methods, "
                    "vulnerabilities, and discovered information."
                )
            }
            
            # Get summary from model
            result = self.summarization_model.invoke(
                [system_message] + formatted_messages
            )
            
            summary = result.content
            
            # Extract topics
            topics = []
            for msg in messages_to_summarize:
                words = msg.text.split()
                for i in range(len(words) - 1):
                    if len(words[i]) > 3 and len(words[i+1]) > 3:
                        potential_topic = f"{words[i]} {words[i+1]}"
                        if potential_topic not in topics:
                            topics.append(potential_topic)
            
            # Limit topics
            topics = topics[:5]
            
            # Create conversation summary
            start_time = min(msg.timestamp for msg in messages_to_summarize)
            end_time = max(msg.timestamp for msg in messages_to_summarize)
            message_ids = [f"msg_{int(msg.timestamp * 1000)}_{hash(str(msg.content)) % 10000}" 
                          for msg in messages_to_summarize]
            
            conversation = Conversation(
                summary=summary,
                start_time=start_time,
                end_time=end_time,
                topics=topics,
                message_ids=message_ids
            )
            
            # Add to conversations
            self.conversations.append(conversation)
            
            # Move summarized messages to long-term memory
            for msg in messages_to_summarize:
                if len(self.long_term_memory) >= self.long_term_limit:
                    # Remove least important message if at capacity
                    self.long_term_memory.sort(key=lambda m: m.importance)
                    self.long_term_memory.pop(0)
                self.long_term_memory.append(msg)
                
        except Exception as e:
            print(f"Error creating conversation summary: {e}")
            # Keep messages in short-term memory
            return


# Helper function to create an advanced memory instance
def create_advanced_memory(session_id=None, persistence_path="./memory"):
    """
    Create and initialize an advanced memory system.
    
    Args:
        session_id: Optional session identifier
        persistence_path: Path for memory persistence
        
    Returns:
        Initialized AdvancedMemory instance
    """
    return AdvancedMemory(
        short_term_limit=20,  # Ensure we have enough short-term memory
        long_term_limit=50,   # Reduced but sufficient for tracking
        working_memory_slots=5,  # Enough for key entities
        session_id=session_id,
        persistence_path=persistence_path
    )


# Helper function to convert between message formats
def convert_to_memory_format(messages):
    """
    Convert LangChain messages to the format expected by the memory system.
    
    Args:
        messages: List of LangChain messages
        
    Returns:
        List of message dictionaries
    """
    result = []
    
    for msg in messages:
        if hasattr(msg, "type"):
            msg_type = msg.type
            content = getattr(msg, "content", "")
        elif hasattr(msg, "role"):
            msg_type = msg.role
            content = getattr(msg, "content", "")
        elif isinstance(msg, dict):
            msg_type = msg.get("type") or msg.get("role", "unknown")
            content = msg.get("content", "")
        else:
            msg_type = "unknown"
            content = str(msg)
            
        result.append({
            "role": msg_type,
            "content": content
        })
        
    return result