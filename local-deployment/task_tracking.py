"""
Task Tracking System for DarkCircuit Agent

This module provides sophisticated task tracking capabilities:
- Tracking of active tasks and their states
- Task dependencies and relationships
- Task prioritization and management
- Agent state awareness across conversations
"""

import os
import re
import time
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set, Union, Callable
from datetime import datetime
from enum import Enum, auto
import hashlib

# Try to import advanced_memory if available
try:
    from advanced_memory import AdvancedMemory
    ADVANCED_MEMORY_AVAILABLE = True
except ImportError:
    ADVANCED_MEMORY_AVAILABLE = False


class TaskStatus(Enum):
    """Enumeration of possible task statuses."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class TaskType(Enum):
    """Enumeration of possible task types."""
    RECONNAISSANCE = "reconnaissance"
    ENUMERATION = "enumeration"
    VULNERABILITY_ANALYSIS = "vulnerability_analysis"
    EXPLOITATION = "exploitation"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    POST_EXPLOITATION = "post_exploitation"
    DOCUMENTATION = "documentation"
    ANALYSIS = "analysis"
    CUSTOM = "custom"


class TaskPriority(Enum):
    """Enumeration of task priorities."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskRelation(Enum):
    """Enumeration of possible task relationships."""
    DEPENDS_ON = "depends_on"
    PARENT_OF = "parent_of"
    RELATED_TO = "related_to"
    BLOCKED_BY = "blocked_by"
    ALTERNATIVE_TO = "alternative_to"


class Task:
    """Represents a task to be tracked by the agent."""
    
    def __init__(self, 
                 title: str, 
                 task_type: Union[TaskType, str],
                 description: Optional[str] = None,
                 priority: Union[TaskPriority, str] = TaskPriority.MEDIUM,
                 expected_duration: Optional[int] = None,  # in seconds
                 task_id: Optional[str] = None):
        """
        Initialize a new task.
        
        Args:
            title: Short title of the task
            task_type: Type of task (reconnaissance, exploitation, etc.)
            description: Optional detailed description of the task
            priority: Task priority (low, medium, high, critical)
            expected_duration: Optional expected time to complete in seconds
            task_id: Optional unique task ID (generated if not provided)
        """
        self.title = title
        
        # Handle string or enum for task_type
        if isinstance(task_type, str):
            try:
                self.task_type = TaskType(task_type)
            except ValueError:
                self.task_type = TaskType.CUSTOM
        else:
            self.task_type = task_type
            
        # Handle string or enum for priority
        if isinstance(priority, str):
            try:
                self.priority = TaskPriority(priority)
            except ValueError:
                self.priority = TaskPriority.MEDIUM
        else:
            self.priority = priority
            
        self.description = description
        self.expected_duration = expected_duration
        self.task_id = task_id or str(uuid.uuid4())
        
        # Task state
        self.status = TaskStatus.PENDING
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.updated_at = self.created_at
        self.progress = 0.0  # 0 to 1.0
        self.attempt_count = 0
        self.notes = []
        
        # Relationships
        self.relations = {
            TaskRelation.DEPENDS_ON: set(),
            TaskRelation.PARENT_OF: set(),
            TaskRelation.RELATED_TO: set(),
            TaskRelation.BLOCKED_BY: set(),
            TaskRelation.ALTERNATIVE_TO: set()
        }
        
        # Results and artifacts
        self.results = {}
        self.artifacts = []
    
    def start(self):
        """Mark the task as in progress."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = time.time()
        self.updated_at = self.started_at
        self.attempt_count += 1
        
    def complete(self, result: Optional[Dict[str, Any]] = None):
        """
        Mark the task as completed.
        
        Args:
            result: Optional dictionary of results from the task
        """
        self.status = TaskStatus.COMPLETED
        self.completed_at = time.time()
        self.updated_at = self.completed_at
        self.progress = 1.0
        if result:
            self.results.update(result)
            
    def fail(self, reason: Optional[str] = None):
        """
        Mark the task as failed.
        
        Args:
            reason: Optional reason for failure
        """
        self.status = TaskStatus.FAILED
        self.updated_at = time.time()
        if reason:
            self.add_note(f"Failed: {reason}")
            
    def block(self, reason: Optional[str] = None, blocked_by_task_id: Optional[str] = None):
        """
        Mark the task as blocked.
        
        Args:
            reason: Optional reason for being blocked
            blocked_by_task_id: Optional ID of task blocking this one
        """
        self.status = TaskStatus.BLOCKED
        self.updated_at = time.time()
        if reason:
            self.add_note(f"Blocked: {reason}")
        if blocked_by_task_id:
            self.add_relation(TaskRelation.BLOCKED_BY, blocked_by_task_id)
            
    def skip(self, reason: Optional[str] = None):
        """
        Mark the task as skipped.
        
        Args:
            reason: Optional reason for skipping
        """
        self.status = TaskStatus.SKIPPED
        self.updated_at = time.time()
        if reason:
            self.add_note(f"Skipped: {reason}")
            
    def update_progress(self, progress: float):
        """
        Update the task progress.
        
        Args:
            progress: Progress value from 0 to 1.0
        """
        self.progress = min(1.0, max(0.0, progress))
        self.updated_at = time.time()
        
    def add_note(self, note: str):
        """
        Add a note to the task.
        
        Args:
            note: Text note to add
        """
        timestamp = time.time()
        formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        self.notes.append({
            "timestamp": timestamp,
            "formatted_time": formatted_time,
            "content": note
        })
        self.updated_at = timestamp
        
    def add_artifact(self, name: str, content: Any, artifact_type: str = "text"):
        """
        Add an artifact to the task.
        
        Args:
            name: Artifact name
            content: Artifact content
            artifact_type: Type of artifact (text, file, etc.)
        """
        self.artifacts.append({
            "name": name,
            "content": content,
            "type": artifact_type,
            "created_at": time.time()
        })
        self.updated_at = time.time()
        
    def add_relation(self, relation_type: Union[TaskRelation, str], related_task_id: str):
        """
        Add a relationship to another task.
        
        Args:
            relation_type: Type of relationship
            related_task_id: ID of the related task
        """
        # Handle string or enum for relation_type
        if isinstance(relation_type, str):
            try:
                relation_type = TaskRelation(relation_type)
            except ValueError:
                relation_type = TaskRelation.RELATED_TO
                
        self.relations[relation_type].add(related_task_id)
        self.updated_at = time.time()
        
    def remove_relation(self, relation_type: Union[TaskRelation, str], related_task_id: str):
        """
        Remove a relationship to another task.
        
        Args:
            relation_type: Type of relationship
            related_task_id: ID of the related task
        """
        # Handle string or enum for relation_type
        if isinstance(relation_type, str):
            try:
                relation_type = TaskRelation(relation_type)
            except ValueError:
                relation_type = TaskRelation.RELATED_TO
                
        if related_task_id in self.relations[relation_type]:
            self.relations[relation_type].remove(related_task_id)
        self.updated_at = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the task
        """
        serialized_relations = {}
        for relation_type, task_ids in self.relations.items():
            serialized_relations[relation_type.value] = list(task_ids)
            
        return {
            "task_id": self.task_id,
            "title": self.title,
            "task_type": self.task_type.value,
            "description": self.description,
            "priority": self.priority.value,
            "expected_duration": self.expected_duration,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "updated_at": self.updated_at,
            "progress": self.progress,
            "attempt_count": self.attempt_count,
            "notes": self.notes,
            "relations": serialized_relations,
            "results": self.results,
            "artifacts": self.artifacts
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """
        Create a task from a dictionary representation.
        
        Args:
            data: Dictionary representation of the task
            
        Returns:
            Reconstructed Task object
        """
        task = cls(
            title=data["title"],
            task_type=data["task_type"],
            description=data.get("description"),
            priority=data.get("priority", "medium"),
            expected_duration=data.get("expected_duration"),
            task_id=data["task_id"]
        )
        
        # Restore state
        task.status = TaskStatus(data["status"])
        task.created_at = data["created_at"]
        task.started_at = data.get("started_at")
        task.completed_at = data.get("completed_at")
        task.updated_at = data["updated_at"]
        task.progress = data["progress"]
        task.attempt_count = data["attempt_count"]
        task.notes = data["notes"]
        
        # Restore relationships
        for relation_name, task_ids in data["relations"].items():
            relation_type = TaskRelation(relation_name)
            task.relations[relation_type] = set(task_ids)
            
        # Restore results and artifacts
        task.results = data["results"]
        task.artifacts = data["artifacts"]
        
        return task


class TaskSet:
    """A collection of related tasks with operations to manage them."""
    
    def __init__(self, name: str, description: Optional[str] = None):
        """
        Initialize a new task set.
        
        Args:
            name: Name of the task set
            description: Optional description of the task set
        """
        self.name = name
        self.description = description
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.tasks: Dict[str, Task] = {}
        self.metadata: Dict[str, Any] = {}
        self.tags: Set[str] = set()
        
    def add_task(self, task: Task) -> str:
        """
        Add a task to the set.
        
        Args:
            task: Task to add
            
        Returns:
            Task ID
        """
        self.tasks[task.task_id] = task
        self.updated_at = time.time()
        return task.task_id
        
    def create_task(self, 
                   title: str, 
                   task_type: Union[TaskType, str],
                   description: Optional[str] = None,
                   priority: Union[TaskPriority, str] = TaskPriority.MEDIUM,
                   expected_duration: Optional[int] = None) -> str:
        """
        Create and add a new task.
        
        Args:
            title: Short title of the task
            task_type: Type of task
            description: Optional detailed description
            priority: Task priority
            expected_duration: Optional expected duration in seconds
            
        Returns:
            Task ID
        """
        task = Task(
            title=title,
            task_type=task_type,
            description=description,
            priority=priority,
            expected_duration=expected_duration
        )
        return self.add_task(task)
        
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.
        
        Args:
            task_id: ID of the task to get
            
        Returns:
            Task object or None if not found
        """
        return self.tasks.get(task_id)
        
    def remove_task(self, task_id: str) -> bool:
        """
        Remove a task from the set.
        
        Args:
            task_id: ID of the task to remove
            
        Returns:
            True if task was removed, False if not found
        """
        if task_id in self.tasks:
            del self.tasks[task_id]
            self.updated_at = time.time()
            return True
        return False
        
    def get_tasks_by_status(self, status: Union[TaskStatus, str]) -> List[Task]:
        """
        Get all tasks with a specific status.
        
        Args:
            status: Status to filter by
            
        Returns:
            List of matching tasks
        """
        # Handle string or enum for status
        if isinstance(status, str):
            try:
                status_enum = TaskStatus(status)
            except ValueError:
                return []
        else:
            status_enum = status
            
        return [task for task in self.tasks.values() if task.status == status_enum]
        
    def get_tasks_by_type(self, task_type: Union[TaskType, str]) -> List[Task]:
        """
        Get all tasks of a specific type.
        
        Args:
            task_type: Type to filter by
            
        Returns:
            List of matching tasks
        """
        # Handle string or enum for task_type
        if isinstance(task_type, str):
            try:
                type_enum = TaskType(task_type)
            except ValueError:
                return []
        else:
            type_enum = task_type
            
        return [task for task in self.tasks.values() if task.task_type == type_enum]
        
    def get_tasks_by_priority(self, priority: Union[TaskPriority, str]) -> List[Task]:
        """
        Get all tasks with a specific priority.
        
        Args:
            priority: Priority to filter by
            
        Returns:
            List of matching tasks
        """
        # Handle string or enum for priority
        if isinstance(priority, str):
            try:
                priority_enum = TaskPriority(priority)
            except ValueError:
                return []
        else:
            priority_enum = priority
            
        return [task for task in self.tasks.values() if task.priority == priority_enum]
        
    def get_next_tasks(self) -> List[Task]:
        """
        Get tasks that are ready to be worked on.
        
        Returns:
            List of tasks that can be started
        """
        pending_tasks = self.get_tasks_by_status(TaskStatus.PENDING)
        
        # Filter out tasks that depend on incomplete tasks
        ready_tasks = []
        for task in pending_tasks:
            dependencies = task.relations[TaskRelation.DEPENDS_ON]
            blocked = task.relations[TaskRelation.BLOCKED_BY]
            
            # Check if all dependencies are completed
            all_dependencies_met = True
            for dep_id in dependencies:
                dep_task = self.get_task(dep_id)
                if dep_task and dep_task.status != TaskStatus.COMPLETED:
                    all_dependencies_met = False
                    break
                    
            # Check if not blocked
            not_blocked = len(blocked) == 0
            
            if all_dependencies_met and not_blocked:
                ready_tasks.append(task)
                
        # Sort by priority
        priority_values = {
            TaskPriority.CRITICAL: 4,
            TaskPriority.HIGH: 3,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 1
        }
        ready_tasks.sort(key=lambda t: priority_values.get(t.priority, 0), reverse=True)
        
        return ready_tasks
        
    def build_task_tree(self) -> Dict[str, Any]:
        """
        Build a hierarchical tree of tasks based on parent-child relationships.
        
        Returns:
            Hierarchical representation of tasks
        """
        # Create a dictionary to store task trees
        task_trees = {}
        root_tasks = []
        
        # Find root tasks (tasks with no parents)
        for task_id, task in self.tasks.items():
            has_parent = False
            for other_id, other_task in self.tasks.items():
                if task_id in other_task.relations[TaskRelation.PARENT_OF]:
                    has_parent = True
                    break
                    
            if not has_parent:
                root_tasks.append(task)
                
        # Build tree for each root task
        for root_task in root_tasks:
            task_trees[root_task.task_id] = self._build_subtree(root_task)
            
        return task_trees
        
    def _build_subtree(self, task: Task) -> Dict[str, Any]:
        """
        Recursively build a subtree for a task.
        
        Args:
            task: The root task for this subtree
            
        Returns:
            Dictionary representing the task and its children
        """
        subtree = task.to_dict()
        children = {}
        
        for child_id in task.relations[TaskRelation.PARENT_OF]:
            child_task = self.get_task(child_id)
            if child_task:
                children[child_id] = self._build_subtree(child_task)
                
        subtree["children"] = children
        return subtree
        
    def create_dependency(self, task_id: str, depends_on_id: str):
        """
        Create a dependency between two tasks.
        
        Args:
            task_id: ID of the dependent task
            depends_on_id: ID of the task it depends on
            
        Returns:
            True if dependency was created successfully
        """
        task = self.get_task(task_id)
        dependency = self.get_task(depends_on_id)
        
        if not task or not dependency:
            return False
            
        task.add_relation(TaskRelation.DEPENDS_ON, depends_on_id)
        self.updated_at = time.time()
        return True
        
    def create_parent_child(self, parent_id: str, child_id: str):
        """
        Create a parent-child relationship between two tasks.
        
        Args:
            parent_id: ID of the parent task
            child_id: ID of the child task
            
        Returns:
            True if relationship was created successfully
        """
        parent = self.get_task(parent_id)
        child = self.get_task(child_id)
        
        if not parent or not child:
            return False
            
        parent.add_relation(TaskRelation.PARENT_OF, child_id)
        self.updated_at = time.time()
        return True
        
    def add_tag(self, tag: str):
        """
        Add a tag to the task set.
        
        Args:
            tag: Tag to add
        """
        self.tags.add(tag)
        self.updated_at = time.time()
        
    def add_metadata(self, key: str, value: Any):
        """
        Add metadata to the task set.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
        self.updated_at = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task set to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the task set
        """
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "metadata": self.metadata,
            "tags": list(self.tags)
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskSet':
        """
        Create a task set from a dictionary representation.
        
        Args:
            data: Dictionary representation of the task set
            
        Returns:
            Reconstructed TaskSet object
        """
        task_set = cls(
            name=data["name"],
            description=data.get("description")
        )
        
        # Restore state
        task_set.created_at = data["created_at"]
        task_set.updated_at = data["updated_at"]
        task_set.metadata = data["metadata"]
        task_set.tags = set(data["tags"])
        
        # Restore tasks
        for task_id, task_data in data["tasks"].items():
            task = Task.from_dict(task_data)
            task_set.tasks[task_id] = task
            
        return task_set


class AgentState:
    """Tracks the overall state of the agent across conversations."""
    
    def __init__(self, agent_id: Optional[str] = None):
        """
        Initialize agent state.
        
        Args:
            agent_id: Optional unique agent identifier
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.last_active = self.created_at
        
        # Runtime state
        self.current_task_id: Optional[str] = None
        self.current_phase: str = "initial"  # reconnaissance, exploitation, etc.
        self.current_target: Optional[Dict[str, Any]] = None
        self.current_context: Dict[str, Any] = {}
        
        # Meta state
        self.session_count = 1
        self.task_sets: Dict[str, TaskSet] = {}
        self.bookmarks: List[Dict[str, Any]] = []
        self.global_metadata: Dict[str, Any] = {}
        self.skills_demonstrated: Set[str] = set()
        
    def update_activity(self):
        """Update the last active timestamp."""
        self.last_active = time.time()
        self.updated_at = self.last_active
        
    def set_current_task(self, task_id: Optional[str] = None):
        """
        Set the current active task.
        
        Args:
            task_id: ID of the current task or None to clear
        """
        self.current_task_id = task_id
        self.updated_at = time.time()
        
    def set_phase(self, phase: str):
        """
        Set the current phase of operation.
        
        Args:
            phase: Current operational phase
        """
        self.current_phase = phase
        self.updated_at = time.time()
        
    def set_target(self, target: Dict[str, Any]):
        """
        Set the current target.
        
        Args:
            target: Target information (IP, hostname, etc.)
        """
        self.current_target = target
        self.updated_at = time.time()
        
    def add_context(self, key: str, value: Any):
        """
        Add information to the current context.
        
        Args:
            key: Context key
            value: Context value
        """
        self.current_context[key] = value
        self.updated_at = time.time()
        
    def clear_context(self, key: Optional[str] = None):
        """
        Clear context data.
        
        Args:
            key: Optional specific key to clear, or all if None
        """
        if key:
            if key in self.current_context:
                del self.current_context[key]
        else:
            self.current_context = {}
        self.updated_at = time.time()
        
    def create_task_set(self, name: str, description: Optional[str] = None) -> str:
        """
        Create a new task set.
        
        Args:
            name: Name of the task set
            description: Optional description
            
        Returns:
            Task set ID
        """
        task_set_id = hashlib.md5(f"{name}_{time.time()}".encode()).hexdigest()[:12]
        task_set = TaskSet(name=name, description=description)
        self.task_sets[task_set_id] = task_set
        self.updated_at = time.time()
        return task_set_id
        
    def get_task_set(self, task_set_id: str) -> Optional[TaskSet]:
        """
        Get a task set by ID.
        
        Args:
            task_set_id: ID of the task set
            
        Returns:
            TaskSet or None if not found
        """
        return self.task_sets.get(task_set_id)
        
    def get_default_task_set(self) -> TaskSet:
        """
        Get or create a default task set.
        
        Returns:
            Default TaskSet
        """
        # Check if default exists
        for task_set_id, task_set in self.task_sets.items():
            if task_set.name == "Default":
                return task_set
                
        # Create default if not found
        task_set_id = self.create_task_set("Default", "Default task set")
        return self.task_sets[task_set_id]
        
    def add_bookmark(self, title: str, content: Any, content_type: str = "text"):
        """
        Add a bookmark to remember important information.
        
        Args:
            title: Bookmark title
            content: Content to bookmark
            content_type: Type of content (text, file, etc.)
        """
        self.bookmarks.append({
            "title": title,
            "content": content,
            "type": content_type,
            "created_at": time.time()
        })
        self.updated_at = time.time()
        
    def add_global_metadata(self, key: str, value: Any):
        """
        Add global metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.global_metadata[key] = value
        self.updated_at = time.time()
        
    def add_skill(self, skill: str):
        """
        Add a demonstrated skill.
        
        Args:
            skill: Skill name
        """
        self.skills_demonstrated.add(skill)
        self.updated_at = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert agent state to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the agent state
        """
        return {
            "agent_id": self.agent_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_active": self.last_active,
            "current_task_id": self.current_task_id,
            "current_phase": self.current_phase,
            "current_target": self.current_target,
            "current_context": self.current_context,
            "session_count": self.session_count,
            "task_sets": {task_set_id: task_set.to_dict() for task_set_id, task_set in self.task_sets.items()},
            "bookmarks": self.bookmarks,
            "global_metadata": self.global_metadata,
            "skills_demonstrated": list(self.skills_demonstrated)
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentState':
        """
        Create an agent state from a dictionary representation.
        
        Args:
            data: Dictionary representation of the agent state
            
        Returns:
            Reconstructed AgentState object
        """
        agent_state = cls(agent_id=data["agent_id"])
        
        # Restore state
        agent_state.created_at = data["created_at"]
        agent_state.updated_at = data["updated_at"]
        agent_state.last_active = data["last_active"]
        agent_state.current_task_id = data["current_task_id"]
        agent_state.current_phase = data["current_phase"]
        agent_state.current_target = data["current_target"]
        agent_state.current_context = data["current_context"]
        agent_state.session_count = data["session_count"]
        agent_state.bookmarks = data["bookmarks"]
        agent_state.global_metadata = data["global_metadata"]
        agent_state.skills_demonstrated = set(data["skills_demonstrated"])
        
        # Restore task sets
        for task_set_id, task_set_data in data["task_sets"].items():
            task_set = TaskSet.from_dict(task_set_data)
            agent_state.task_sets[task_set_id] = task_set
            
        return agent_state


class TaskTracker:
    """
    Main interface for task tracking and agent state management.
    Combines task tracking with memory management and state persistence.
    """
    
    def __init__(self, 
                 agent_id: Optional[str] = None,
                 persistence_path: Optional[str] = None,
                 memory: Optional[Any] = None):
        """
        Initialize a task tracker.
        
        Args:
            agent_id: Optional unique agent identifier
            persistence_path: Optional path for state persistence
            memory: Optional AdvancedMemory instance
        """
        self.agent_state = AgentState(agent_id)
        self.memory = memory
        self.persistence_path = persistence_path
        self.default_task_set = self.agent_state.get_default_task_set()
        self.session_tasks: List[str] = []  # Task IDs created in current session
        
        # Load state if it exists
        if persistence_path and agent_id:
            self.load_state()
            
        # Set current session
        self.agent_state.session_count += 1
        self.agent_state.update_activity()
        
    def create_task(self, 
                   title: str, 
                   task_type: Union[TaskType, str],
                   description: Optional[str] = None,
                   priority: Union[TaskPriority, str] = TaskPriority.MEDIUM,
                   task_set_id: Optional[str] = None) -> str:
        """
        Create a new task.
        
        Args:
            title: Task title
            task_type: Type of task
            description: Optional description
            priority: Task priority
            task_set_id: Optional task set ID (uses default if None)
            
        Returns:
            Task ID
        """
        task_set = self.default_task_set
        if task_set_id:
            task_set_obj = self.agent_state.get_task_set(task_set_id)
            if task_set_obj:
                task_set = task_set_obj
                
        task_id = task_set.create_task(
            title=title,
            task_type=task_type,
            description=description,
            priority=priority
        )
        
        self.session_tasks.append(task_id)
        self.agent_state.update_activity()
        
        return task_id
        
    def start_task(self, task_id: str) -> bool:
        """
        Start a task and set it as the current task.
        
        Args:
            task_id: ID of the task to start
            
        Returns:
            True if task was started successfully
        """
        for task_set_id, task_set in self.agent_state.task_sets.items():
            task = task_set.get_task(task_id)
            if task:
                task.start()
                self.agent_state.set_current_task(task_id)
                self.agent_state.update_activity()
                return True
                
        return False
        
    def complete_task(self, task_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """
        Complete a task.
        
        Args:
            task_id: ID of the task to complete
            result: Optional task results
            
        Returns:
            True if task was completed successfully
        """
        for task_set_id, task_set in self.agent_state.task_sets.items():
            task = task_set.get_task(task_id)
            if task:
                task.complete(result)
                if self.agent_state.current_task_id == task_id:
                    self.agent_state.set_current_task(None)
                self.agent_state.update_activity()
                return True
                
        return False
        
    def fail_task(self, task_id: str, reason: Optional[str] = None) -> bool:
        """
        Mark a task as failed.
        
        Args:
            task_id: ID of the task to mark as failed
            reason: Optional reason for failure
            
        Returns:
            True if task was marked as failed successfully
        """
        for task_set_id, task_set in self.agent_state.task_sets.items():
            task = task_set.get_task(task_id)
            if task:
                task.fail(reason)
                if self.agent_state.current_task_id == task_id:
                    self.agent_state.set_current_task(None)
                self.agent_state.update_activity()
                return True
                
        return False
        
    def add_task_note(self, task_id: str, note: str) -> bool:
        """
        Add a note to a task.
        
        Args:
            task_id: ID of the task
            note: Note to add
            
        Returns:
            True if note was added successfully
        """
        for task_set_id, task_set in self.agent_state.task_sets.items():
            task = task_set.get_task(task_id)
            if task:
                task.add_note(note)
                self.agent_state.update_activity()
                return True
                
        return False
        
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a task by ID.
        
        Args:
            task_id: ID of the task to get
            
        Returns:
            Task dictionary or None if not found
        """
        for task_set_id, task_set in self.agent_state.task_sets.items():
            task = task_set.get_task(task_id)
            if task:
                return task.to_dict()
                
        return None
        
    def get_current_task(self) -> Optional[Dict[str, Any]]:
        """
        Get the current active task.
        
        Returns:
            Current task dictionary or None if no active task
        """
        task_id = self.agent_state.current_task_id
        if task_id:
            return self.get_task(task_id)
        return None
        
    def get_next_tasks(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the next tasks to work on.
        
        Args:
            limit: Maximum number of tasks to return
            
        Returns:
            List of task dictionaries
        """
        all_next_tasks = []
        
        for task_set_id, task_set in self.agent_state.task_sets.items():
            next_tasks = task_set.get_next_tasks()
            all_next_tasks.extend([(task_set_id, task) for task in next_tasks])
            
        # Sort by priority
        priority_values = {
            TaskPriority.CRITICAL: 4,
            TaskPriority.HIGH: 3,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 1
        }
        all_next_tasks.sort(key=lambda t: priority_values.get(t[1].priority, 0), reverse=True)
        
        # Return limited number of tasks
        return [task.to_dict() for _, task in all_next_tasks[:limit]]
        
    def get_session_tasks(self) -> List[Dict[str, Any]]:
        """
        Get tasks created in the current session.
        
        Returns:
            List of task dictionaries
        """
        session_task_dicts = []
        
        for task_id in self.session_tasks:
            task_dict = self.get_task(task_id)
            if task_dict:
                session_task_dicts.append(task_dict)
                
        return session_task_dicts
        
    def set_target(self, ip: str, hostname: Optional[str] = None, additional_info: Optional[Dict[str, Any]] = None):
        """
        Set the current target information.
        
        Args:
            ip: Target IP address
            hostname: Optional hostname
            additional_info: Optional additional target information
        """
        target = {
            "ip": ip,
            "hostname": hostname,
            "additional_info": additional_info or {}
        }
        self.agent_state.set_target(target)
        
        # Create target-specific task set if it doesn't exist
        target_name = hostname or ip
        target_task_set_exists = False
        
        for task_set_id, task_set in self.agent_state.task_sets.items():
            if task_set.name == f"Target: {target_name}":
                target_task_set_exists = True
                break
                
        if not target_task_set_exists:
            self.agent_state.create_task_set(
                name=f"Target: {target_name}",
                description=f"Tasks related to target {target_name} ({ip})"
            )
        
    def set_phase(self, phase: str):
        """
        Set the current operational phase.
        
        Args:
            phase: Phase name
        """
        self.agent_state.set_phase(phase)
        
    def add_bookmark(self, title: str, content: Any):
        """
        Add a bookmark for important information.
        
        Args:
            title: Bookmark title
            content: Content to bookmark
        """
        self.agent_state.add_bookmark(title, content)
        
    def save_state(self, file_path: Optional[str] = None) -> Optional[str]:
        """
        Save the current state to disk.
        
        Args:
            file_path: Optional explicit file path
            
        Returns:
            Path where state was saved or None if unable to save
        """
        if file_path is None:
            if self.persistence_path is None:
                return None
                
            if not os.path.exists(self.persistence_path):
                try:
                    os.makedirs(self.persistence_path, exist_ok=True)
                except Exception as e:
                    print(f"Error creating persistence directory: {e}")
                    return None
                    
            file_path = os.path.join(
                self.persistence_path,
                f"agent_state_{self.agent_state.agent_id}.json"
            )
            
        try:
            # Convert to serializable format
            state_dict = self.agent_state.to_dict()
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(state_dict, f, indent=2)
            return file_path
        except Exception as e:
            print(f"Error saving agent state: {e}")
            
            # Try a minimal version if full serialization fails
            try:
                minimal_state = {
                    "agent_id": self.agent_state.agent_id,
                    "current_phase": self.agent_state.current_phase,
                    "current_target": self.agent_state.current_target,
                    "session_count": self.agent_state.session_count,
                    "created_at": self.agent_state.created_at,
                    "updated_at": self.agent_state.updated_at,
                    "last_active": self.agent_state.last_active
                }
                
                backup_path = file_path.replace('.json', '_minimal.json')
                with open(backup_path, 'w') as f:
                    json.dump(minimal_state, f)
                return backup_path
            except Exception as inner_e:
                print(f"Error saving minimal agent state: {inner_e}")
                return None
            
    def load_state(self, file_path: Optional[str] = None) -> bool:
        """
        Load state from disk.
        
        Args:
            file_path: Optional explicit file path
            
        Returns:
            True if state was loaded successfully
        """
        if file_path is None:
            if self.persistence_path is None:
                return False
                
            file_path = os.path.join(
                self.persistence_path,
                f"agent_state_{self.agent_state.agent_id}.json"
            )
            
            # Check for minimal version if main file doesn't exist
            if not os.path.exists(file_path):
                minimal_path = file_path.replace('.json', '_minimal.json')
                if os.path.exists(minimal_path):
                    file_path = minimal_path
                else:
                    return False
            
        if not os.path.exists(file_path):
            return False
            
        try:
            with open(file_path, 'r') as f:
                state_data = json.load(f)
            
            # Handle minimal state format
            if 'agent_id' in state_data and len(state_data) < 10:
                # Just restore the basic agent info
                self.agent_state.agent_id = state_data.get("agent_id", self.agent_state.agent_id)
                self.agent_state.current_phase = state_data.get("current_phase", self.agent_state.current_phase)
                self.agent_state.current_target = state_data.get("current_target", self.agent_state.current_target)
                self.agent_state.session_count = state_data.get("session_count", 0) + 1
                self.agent_state.update_activity()
                return True
            
            # Full state restoration
            try:
                self.agent_state = AgentState.from_dict(state_data)
                self.default_task_set = self.agent_state.get_default_task_set()
                return True
            except Exception as parse_error:
                print(f"Error parsing agent state: {parse_error}")
                # Create a new state with the same ID
                self.agent_state = AgentState(agent_id=state_data.get("agent_id", self.agent_state.agent_id))
                self.default_task_set = self.agent_state.get_default_task_set()
                return False
                
        except Exception as e:
            print(f"Error loading agent state: {e}")
            return False
            
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current agent state.
        
        Returns:
            Dictionary with agent state summary
        """
        # Count tasks by status
        tasks_by_status = {}
        tasks_by_type = {}
        total_tasks = 0
        
        for task_set_id, task_set in self.agent_state.task_sets.items():
            for task_id, task in task_set.tasks.items():
                status_key = task.status.value
                type_key = task.task_type.value
                
                if status_key not in tasks_by_status:
                    tasks_by_status[status_key] = 0
                tasks_by_status[status_key] += 1
                
                if type_key not in tasks_by_type:
                    tasks_by_type[type_key] = 0
                tasks_by_type[type_key] += 1
                
                total_tasks += 1
                
        # Get current task info
        current_task = self.get_current_task()
        
        # Create summary
        return {
            "agent_id": self.agent_state.agent_id,
            "session_count": self.agent_state.session_count,
            "current_phase": self.agent_state.current_phase,
            "current_target": self.agent_state.current_target,
            "current_task": current_task,
            "task_sets": len(self.agent_state.task_sets),
            "total_tasks": total_tasks,
            "tasks_by_status": tasks_by_status,
            "tasks_by_type": tasks_by_type,
            "bookmarks": len(self.agent_state.bookmarks),
            "skills_demonstrated": list(self.agent_state.skills_demonstrated)
        }
        
    def get_task_timeline(self) -> List[Dict[str, Any]]:
        """
        Get a chronological timeline of all tasks.
        
        Returns:
            List of task dictionaries sorted by creation time
        """
        timeline = []
        
        for task_set_id, task_set in self.agent_state.task_sets.items():
            for task_id, task in task_set.tasks.items():
                timeline.append({
                    "task_id": task_id,
                    "task_set_id": task_set_id,
                    "task_set_name": task_set.name,
                    "title": task.title,
                    "status": task.status.value,
                    "created_at": task.created_at,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at,
                    "priority": task.priority.value,
                    "task_type": task.task_type.value
                })
                
        # Sort by creation time
        timeline.sort(key=lambda t: t["created_at"])
        
        return timeline
        
    def integrate_with_memory(self, memory_obj: Any):
        """
        Integrate with a memory system.
        
        Args:
            memory_obj: Memory system object
        """
        self.memory = memory_obj


# Helper function to create a task tracker
def create_task_tracker(agent_id: Optional[str] = None, 
                       persistence_path: Optional[str] = "./agent_state",
                       memory: Optional[Any] = None) -> TaskTracker:
    """
    Create and initialize a task tracker.
    
    Args:
        agent_id: Optional agent identifier
        persistence_path: Optional persistence path
        memory: Optional memory object
        
    Returns:
        Initialized TaskTracker
    """
    return TaskTracker(
        agent_id=agent_id,
        persistence_path=persistence_path,
        memory=memory
    )