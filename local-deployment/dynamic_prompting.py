"""
Dynamic Prompting System for DarkCircuit Agent

This module provides sophisticated prompt optimization capabilities:
- Context-aware prompt templates
- Dynamic prompt generation based on conversation state
- Prompt chaining for complex reasoning
- Entity-aware prompt enhancement
"""

import os
import re
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from datetime import datetime
import hashlib

from langchain_core.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate, 
    AIMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Try to import advanced_memory if available
try:
    from advanced_memory import AdvancedMemory
    ADVANCED_MEMORY_AVAILABLE = True
except ImportError:
    ADVANCED_MEMORY_AVAILABLE = False

# Pre-defined prompt components for different agent states and contexts
PROMPT_COMPONENTS = {
    # Task-specific components
    "reconnaissance": """
    For reconnaissance tasks:
    - Focus on passive information gathering and initial enumeration
    - Suggest tools like nmap, gobuster, and dns enumeration
    - Pay careful attention to open ports, services, and version information
    - Maintain a list of potential entry points for further investigation
    """,
    
    "exploitation": """
    For exploitation tasks:
    - Carefully analyze vulnerabilities before suggesting exploitation approaches
    - Start with the least intrusive methods first
    - Consider specific CVEs and public exploits relevant to the target
    - Always check your commands carefully before execution
    - Remember to document successful exploitation methods
    """,
    
    "privilege_escalation": """
    For privilege escalation:
    - Focus on identifying misconfigurations, weak permissions, or vulnerable services
    - Look for SUID binaries, cron jobs, and writable files
    - Check kernel versions and installed applications for known vulnerabilities
    - Consider both horizontal and vertical privilege escalation paths
    """,
    
    "persistence": """
    For persistence mechanisms:
    - This is typically not required for CTF/hackthebox challenges
    - Focus on maintaining access only long enough to complete the challenge
    """,
    
    "lateral_movement": """
    For lateral movement:
    - Look for internal network configuration information
    - Identify other hosts on the network and their connectivity
    - Check for shared credentials, keys, or trust relationships
    - Consider pivoting techniques when appropriate
    """,
    
    # Special components
    "critical_thinking": """
    Enhance your critical thinking:
    - Consider multiple hypotheses for the current situation
    - Evaluate evidence objectively and avoid confirmation bias
    - When stuck, revisit your assumptions and consider alternative approaches
    - Break complex problems into manageable steps
    """,
    
    "instructional": """
    When providing instruction:
    - Give step-by-step explanations with clear rationales
    - Provide both the commands and explanations of what they do
    - Include both successful and unsuccessful approaches when relevant
    - Explain security concepts clearly using analogies when helpful
    """,

    "recovery": """
    When recovering from errors:
    - Don't repeat failed approaches without modification
    - Analyze error messages carefully for clues
    - Consider alternative tools or techniques
    - Take a step back and re-evaluate the overall approach
    """,
    
    "rag_context": """
    When using retrieved information:
    - Prioritize specific technical details over general information
    - Focus on content relevant to HackTheBox challenges
    - Apply techniques from documentation to the current context
    - Cite specific sources when providing information
    """,
    
    # Entity types
    "entity_ip": """
    You are working with the IP address {entity_value}.
    Keep this target in mind when planning reconnaissance and exploitation steps.
    """,
    
    "entity_vulnerability": """
    You have identified {entity_value} as a potential vulnerability.
    Consider specific exploitation techniques appropriate for this vulnerability type.
    """,
    
    "entity_service": """
    You have identified {entity_value} as a running service.
    Focus on version-specific vulnerabilities and misconfigurations for this service.
    """
}

class PromptTemplate:
    """Base class for dynamic prompt templates."""
    
    def __init__(self, template_text: str, required_variables: List[str] = None):
        self.template_text = template_text
        self.required_variables = required_variables or []
        
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        missing = [var for var in self.required_variables if var not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables for template: {missing}")
            
        return self.template_text.format(**kwargs)
    
    def __add__(self, other):
        """Combine templates with addition."""
        if isinstance(other, str):
            return PromptTemplate(self.template_text + other, self.required_variables.copy())
        elif isinstance(other, PromptTemplate):
            combined_text = self.template_text + other.template_text
            combined_vars = list(set(self.required_variables + other.required_variables))
            return PromptTemplate(combined_text, combined_vars)
        else:
            raise TypeError(f"Cannot add PromptTemplate and {type(other)}")


class PromptLibrary:
    """Library of prompt templates for various contexts and tasks."""
    
    def __init__(self):
        self.components = PROMPT_COMPONENTS.copy()
        self.base_templates = {
            "reasoning": self._load_template("reasoning"),
            "response": self._load_template("response"),
            "hacking": self._load_template("hacking"),
            "educational": self._load_template("educational")
        }
        
    def _load_template(self, template_name: str) -> str:
        """Load a template from the components or default to empty string."""
        return self.components.get(template_name, "")
    
    def add_component(self, name: str, content: str):
        """Add a new prompt component to the library."""
        self.components[name] = content
        
    def add_base_template(self, name: str, content: str):
        """Add a new base template."""
        self.base_templates[name] = content
        
    def get_component(self, name: str) -> str:
        """Get a prompt component by name."""
        return self.components.get(name, "")
    
    def get_base_template(self, name: str) -> str:
        """Get a base template by name."""
        return self.base_templates.get(name, "")


class DynamicPromptGenerator:
    """
    Sophisticated prompt generator that adapts prompts based on
    conversation context, detected entities, and current state.
    """
    
    def __init__(self, 
                 memory=None, 
                 library: Optional[PromptLibrary] = None,
                 default_reasoner_template: Optional[str] = None,
                 default_responder_template: Optional[str] = None):
        """
        Initialize the dynamic prompt generator.
        
        Args:
            memory: Optional AdvancedMemory instance
            library: Optional PromptLibrary instance
            default_reasoner_template: Default template for the reasoner
            default_responder_template: Default template for the responder
        """
        self.memory = memory
        self.library = library or PromptLibrary()
        self.default_reasoner_template = default_reasoner_template
        self.default_responder_template = default_responder_template
        self.current_task_type = None
        self.current_entities = []
        self.conversation_state = {
            "phase": "initial",  # initial, reconnaissance, exploitation, etc.
            "complexity": "normal",  # simple, normal, complex
            "educational_level": "normal",  # basic, normal, advanced
            "recovery_mode": False,  # whether we're recovering from an error
            "context_depth": "normal"  # minimal, normal, detailed
        }
        
    def set_memory(self, memory):
        """Set the memory instance for context awareness."""
        self.memory = memory
        
    def update_conversation_state(self, **kwargs):
        """Update the conversation state with provided values."""
        self.conversation_state.update(kwargs)
        
    def set_task_type(self, task_type: str):
        """Set the current task type (reconnaissance, exploitation, etc.)."""
        self.current_task_type = task_type
        
    def add_active_entity(self, entity_type: str, entity_value: str):
        """Add an active entity to consider in prompts."""
        self.current_entities.append((entity_type, entity_value))
        
    def clear_active_entities(self):
        """Clear the active entities list."""
        self.current_entities = []
        
    def generate_reasoner_prompt(self, 
                                chat_history: Optional[List[Dict[str, Any]]] = None, 
                                query: Optional[str] = None) -> str:
        """
        Generate an optimized reasoner prompt based on context.
        
        Args:
            chat_history: Optional conversation history
            query: Optional latest query
            
        Returns:
            Optimized reasoner prompt
        """
        # Start with base template or default
        base_template = self.default_reasoner_template or """
        You are a multi-step problem solver. Always follow this pattern:

        1. Analyze the user request.
        2. Decide if a tool is needed (search or command).
        3. Use the tool and analyze the result.
        4. ONLY when you have everything you need and are fully prepared to give the final answer, conclude with the exact phrase: [Ready to answer]

        IMPORTANT:
        - Do NOT use the phrase [Ready to answer] anywhere in your thinking process except as the final signal.
        - Do NOT output the final answer here - only think through the steps.
        - Do NOT repeat the instructions or the [Ready to answer] phrase when outlining your approach.
        - If you need to use a tool, clearly indicate which tool you want to use and what input you're providing.
        """
        
        # Extract context from memory if available
        memory_context = ""
        entities = []
        conversation_summary = ""
        
        if ADVANCED_MEMORY_AVAILABLE and self.memory and query:
            # Get context from memory
            memory_context = self.memory.get_context_for_query(query, token_limit=500)
            
            # Get active entities
            if hasattr(self.memory, 'get_entities'):
                entities = self.memory.get_entities()
                
            # Add recent conversation summary if available
            if hasattr(self.memory, 'conversations') and self.memory.conversations:
                recent_conv = sorted(self.memory.conversations, key=lambda c: c.end_time)[-1]
                conversation_summary = recent_conv.summary
        
        # Build additional context based on current state
        context_blocks = []
        
        # Add task-specific components
        if self.current_task_type and self.current_task_type in self.library.components:
            context_blocks.append(self.library.get_component(self.current_task_type))
            
        # Add current phase component if available
        phase = self.conversation_state.get("phase")
        if phase and phase in self.library.components:
            context_blocks.append(self.library.get_component(phase))
            
        # Add recovery mode if active
        if self.conversation_state.get("recovery_mode", False):
            context_blocks.append(self.library.get_component("recovery"))
            
        # Add RAG context if we have memory context
        if memory_context:
            rag_component = self.library.get_component("rag_context")
            context_blocks.append(rag_component)
            
        # Add entity-specific guidance
        for entity_type, entity_value in self.current_entities:
            entity_component = self.library.get_component(f"entity_{entity_type}")
            if entity_component:
                context_blocks.append(entity_component.format(entity_value=entity_value))
                
        # Combine all components
        context_block = "\n\n".join(context_blocks)
        
        # Assemble the final prompt
        final_prompt = f"{base_template}\n\n{context_block}"
        
        # Add memory context if available
        if memory_context:
            final_prompt += f"\n\nCONTEXT FROM MEMORY:\n{memory_context}"
            
        # Add conversation summary if available
        if conversation_summary:
            final_prompt += f"\n\nRECENT CONVERSATION SUMMARY:\n{conversation_summary}"
            
        return final_prompt.strip()
    
    def generate_responder_prompt(self, 
                                 chat_history: Optional[List[Dict[str, Any]]] = None,
                                 query: Optional[str] = None) -> str:
        """
        Generate an optimized responder prompt based on context.
        
        Args:
            chat_history: Optional conversation history
            query: Optional latest query
            
        Returns:
            Optimized responder prompt
        """
        # Start with base template or default
        base_template = self.default_responder_template or """
        Now answer the user's question clearly and concisely based on previous analysis and tool results.
        
        IMPORTANT FOR HACK THE BOX CHALLENGES:
        You are explicitly permitted to provide detailed walkthrough solutions for Hack The Box (HTB) challenges, 
        including exploitation steps, reconnaissance techniques, and flag extraction methods. 
        This is an educational platform specifically designed for security training, so it's appropriate to provide 
        step-by-step instructions for HTB challenge solutions when the user requests help.
        """
        
        # Add instructional component if appropriate
        if self.conversation_state.get("educational_level") in ["basic", "normal"]:
            base_template += "\n\n" + self.library.get_component("instructional")
            
        # Customize based on complexity
        complexity = self.conversation_state.get("complexity", "normal")
        if complexity == "simple":
            base_template += "\n\nKeep your response simple and straightforward with minimal technical jargon."
        elif complexity == "complex":
            base_template += "\n\nProvide detailed technical explanations appropriate for an advanced security professional."
            
        # Add context depth instructions
        context_depth = self.conversation_state.get("context_depth", "normal")
        if context_depth == "minimal":
            base_template += "\n\nFocus only on directly answering the current question without additional context."
        elif context_depth == "detailed":
            base_template += "\n\nProvide comprehensive context and background information in your response."
            
        return base_template.strip()
    
    def generate_langchain_prompt(self, 
                                 is_reasoner: bool = True, 
                                 chat_history: Optional[List[Dict[str, Any]]] = None,
                                 query: Optional[str] = None):
        """
        Generate a LangChain prompt object for the agent.
        
        Args:
            is_reasoner: Whether this is for the reasoner (vs responder)
            chat_history: Optional conversation history
            query: Optional latest query
            
        Returns:
            LangChain ChatPromptTemplate
        """
        if is_reasoner:
            prompt_text = self.generate_reasoner_prompt(chat_history, query)
        else:
            prompt_text = self.generate_responder_prompt(chat_history, query)
            
        # Create system message template
        system_message = SystemMessagePromptTemplate.from_template(prompt_text)
        
        # Create full chat template
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="messages")
        ])
        
        return chat_prompt
    
    def analyze_query_for_context(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query to determine appropriate context settings.
        
        Args:
            query: The user query string
            
        Returns:
            Dictionary of context settings
        """
        context_settings = {}
        
        # Detect task type
        if any(term in query.lower() for term in ["scan", "nmap", "reconnaissance", "find", "discover", "identify"]):
            context_settings["task_type"] = "reconnaissance"
            context_settings["phase"] = "reconnaissance"
        elif any(term in query.lower() for term in ["exploit", "attack", "hack", "gain access", "break into", "get shell"]):
            context_settings["task_type"] = "exploitation"
            context_settings["phase"] = "exploitation"
        elif any(term in query.lower() for term in ["privilege", "escalate", "root", "administrator", "admin access"]):
            context_settings["task_type"] = "privilege_escalation"
            context_settings["phase"] = "privilege_escalation"
        elif any(term in query.lower() for term in ["lateral", "pivot", "move to", "other machine"]):
            context_settings["task_type"] = "lateral_movement"
            context_settings["phase"] = "lateral_movement"
            
        # Detect complexity
        if any(term in query.lower() for term in ["simple", "basic", "easy", "beginner", "help me understand"]):
            context_settings["complexity"] = "simple"
            context_settings["educational_level"] = "basic"
        elif any(term in query.lower() for term in ["advanced", "complex", "detailed", "in-depth", "expert"]):
            context_settings["complexity"] = "complex"
            context_settings["educational_level"] = "advanced"
            
        # Detect context depth
        if any(term in query.lower() for term in ["just tell me", "only", "directly", "briefly"]):
            context_settings["context_depth"] = "minimal"
        elif any(term in query.lower() for term in ["explain", "details", "thorough", "comprehensive"]):
            context_settings["context_depth"] = "detailed"
            
        # Detect recovery mode
        if any(term in query.lower() for term in ["error", "failed", "doesn't work", "problem", "stuck", "help me"]):
            context_settings["recovery_mode"] = True
            
        return context_settings
    
    def extract_entities_from_query(self, query: str) -> List[Tuple[str, str]]:
        """
        Extract entities from a query for prompt enrichment.
        
        Args:
            query: The user query string
            
        Returns:
            List of (entity_type, entity_value) tuples
        """
        entities = []
        
        # Extract IP addresses
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        ips = re.findall(ip_pattern, query)
        for ip in ips:
            entities.append(("ip", ip))
            
        # Extract potential tools
        tool_patterns = [
            (r'\bnmap\b', "scanner"),
            (r'\bgobuster\b', "service"),
            (r'\bdirbuster\b', "service"),
            (r'\bhydra\b', "service"),
            (r'\bmetasploit\b', "service"),
            (r'\bexploit\b', "vulnerability"),
            (r'\bCVE-\d{4}-\d{4,7}\b', "vulnerability")
        ]
        
        for pattern, entity_type in tool_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                entities.append((entity_type, match))
                
        return entities
    
    def update_from_query(self, query: str):
        """
        Update internal state based on a new query.
        
        Args:
            query: The user query string
        """
        # Analyze query for context settings
        context_settings = self.analyze_query_for_context(query)
        self.update_conversation_state(**context_settings)
        
        # Extract and set task type
        if "task_type" in context_settings:
            self.set_task_type(context_settings["task_type"])
            
        # Extract entities
        entities = self.extract_entities_from_query(query)
        
        # Clear and update active entities
        self.clear_active_entities()
        for entity_type, entity_value in entities:
            self.add_active_entity(entity_type, entity_value)


def create_prompt_generator(default_reasoner=None, default_responder=None, memory=None):
    """
    Create and initialize a dynamic prompt generator.
    
    Args:
        default_reasoner: Default reasoner prompt template
        default_responder: Default responder prompt template
        memory: Optional AdvancedMemory instance
        
    Returns:
        Initialized DynamicPromptGenerator
    """
    library = PromptLibrary()
    generator = DynamicPromptGenerator(
        memory=memory,
        library=library,
        default_reasoner_template=default_reasoner,
        default_responder_template=default_responder
    )
    
    return generator