"""
Learning System for DarkCircuit Agent

This module provides learning capabilities for the DarkCircuit agent by:
1. Tracking successful command sequences
2. Recording exploitation paths that led to flags or successful outcomes
3. Building a knowledge base of effective techniques
4. Providing retrieval of relevant past successful approaches
"""

import os
import json
import time
import re
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import sqlite3
import hashlib

class CommandSequence:
    """A sequence of commands that were executed together as part of a task"""
    
    def __init__(self, 
                 name: str = "",
                 description: str = "",
                 commands: List[Dict[str, Any]] = None,
                 tags: List[str] = None,
                 success_level: int = 0,
                 target_type: str = "unknown"):
        self.id = hashlib.md5(f"{name}_{time.time()}".encode()).hexdigest()[:12]
        self.name = name
        self.description = description
        self.commands = commands or []
        self.tags = tags or []
        self.success_level = success_level  # 0=unknown, 1=failed, 2=partial success, 3=complete success
        self.created_at = time.time()
        self.target_type = target_type  # e.g., "web", "linux", "windows"
        self.outcome = ""
        self.flag_found = False
        
    def add_command(self, 
                   command: str, 
                   result: str, 
                   success: bool = False, 
                   importance: int = 1):
        """Add a command to the sequence"""
        self.commands.append({
            "command": command,
            "result": result,
            "success": success,
            "importance": importance,
            "timestamp": time.time()
        })
        
    def mark_success(self, success_level: int, outcome: str = ""):
        """Mark the success level of this command sequence"""
        self.success_level = success_level
        if outcome:
            self.outcome = outcome
            
    def mark_flag_found(self, flag: str = ""):
        """Mark that a flag was found in this sequence"""
        self.flag_found = True
        if flag and flag not in self.outcome:
            self.outcome = f"{self.outcome}\nFlag: {flag}".strip()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "commands": self.commands,
            "tags": self.tags,
            "success_level": self.success_level,
            "created_at": self.created_at,
            "target_type": self.target_type,
            "outcome": self.outcome,
            "flag_found": self.flag_found
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommandSequence':
        """Create from dictionary"""
        seq = cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            commands=data.get("commands", []),
            tags=data.get("tags", []),
            success_level=data.get("success_level", 0),
            target_type=data.get("target_type", "unknown")
        )
        seq.id = data.get("id", seq.id)
        seq.created_at = data.get("created_at", seq.created_at)
        seq.outcome = data.get("outcome", "")
        seq.flag_found = data.get("flag_found", False)
        return seq


class TechniqueLibrary:
    """A library of techniques that have been learned"""
    
    def __init__(self, 
                 db_path: str = "learning.db",
                 persistence_path: str = "./learning_data"):
        self.db_path = os.path.join(persistence_path, db_path)
        self.persistence_path = persistence_path
        
        # Create persistence directory if it doesn't exist
        if not os.path.exists(persistence_path):
            os.makedirs(persistence_path, exist_ok=True)
            
        # Initialize in-memory caches
        self.command_sequences = []  # All known command sequences
        self.successful_commands = defaultdict(int)  # Command -> success count
        self.technique_tags = defaultdict(set)  # tag -> sequence ids
        self.target_type_index = defaultdict(set)  # target type -> sequence ids
        
        # Command pattern index
        self.command_pattern_index = {}  # command pattern -> [sequence ids]
        
        # Initialize database
        self._init_database()
        
        # Load data
        self._load_data()
        
    def _init_database(self):
        """Initialize the SQLite database for learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS command_sequences (
            id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            success_level INTEGER,
            target_type TEXT,
            flag_found INTEGER,
            created_at REAL
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS command_stats (
            command TEXT PRIMARY KEY,
            success_count INTEGER,
            usage_count INTEGER,
            avg_importance REAL
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS technique_tags (
            tag TEXT,
            sequence_id TEXT,
            PRIMARY KEY (tag, sequence_id)
        )
        """)
        
        conn.commit()
        conn.close()
        
    def _load_data(self):
        """Load data from database into memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load command sequences
        cursor.execute("SELECT data FROM command_sequences")
        for (data,) in cursor.fetchall():
            try:
                seq_dict = json.loads(data)
                sequence = CommandSequence.from_dict(seq_dict)
                self.command_sequences.append(sequence)
                
                # Index by tags
                for tag in sequence.tags:
                    self.technique_tags[tag].add(sequence.id)
                
                # Index by target type
                self.target_type_index[sequence.target_type].add(sequence.id)
                
                # Build command pattern index
                for cmd_entry in sequence.commands:
                    cmd = cmd_entry.get("command", "")
                    pattern = self._extract_command_pattern(cmd)
                    if pattern:
                        if pattern not in self.command_pattern_index:
                            self.command_pattern_index[pattern] = set()
                        self.command_pattern_index[pattern].add(sequence.id)
            except Exception as e:
                print(f"Error loading sequence: {e}")
                
        # Load command stats
        cursor.execute("SELECT command, success_count FROM command_stats")
        for command, success_count in cursor.fetchall():
            self.successful_commands[command] = success_count
            
        conn.close()
                
    def add_command_sequence(self, sequence: CommandSequence):
        """Add a command sequence to the library"""
        # Check for duplicates
        existing_ids = {seq.id for seq in self.command_sequences}
        if sequence.id in existing_ids:
            # Update existing sequence
            for i, seq in enumerate(self.command_sequences):
                if seq.id == sequence.id:
                    self.command_sequences[i] = sequence
                    break
        else:
            # Add new sequence
            self.command_sequences.append(sequence)
            
        # Update indexes
        for tag in sequence.tags:
            self.technique_tags[tag].add(sequence.id)
            
        self.target_type_index[sequence.target_type].add(sequence.id)
        
        # Index command patterns
        for cmd_entry in sequence.commands:
            cmd = cmd_entry.get("command", "")
            success = cmd_entry.get("success", False)
            
            # Update successful commands count
            if success:
                self.successful_commands[cmd] += 1
                
            # Update command pattern index
            pattern = self._extract_command_pattern(cmd)
            if pattern:
                if pattern not in self.command_pattern_index:
                    self.command_pattern_index[pattern] = set()
                self.command_pattern_index[pattern].add(sequence.id)
        
        # Save to database
        self._save_sequence(sequence)
        
    def _save_sequence(self, sequence: CommandSequence):
        """Save a sequence to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert to JSON
        data_json = json.dumps(sequence.to_dict())
        
        # Save/update the sequence
        cursor.execute(
            """
            INSERT OR REPLACE INTO command_sequences 
            (id, data, success_level, target_type, flag_found, created_at) 
            VALUES (?, ?, ?, ?, ?, ?)
            """, 
            (sequence.id, data_json, sequence.success_level, 
             sequence.target_type, 1 if sequence.flag_found else 0, sequence.created_at)
        )
        
        # Save technique tags
        cursor.execute("DELETE FROM technique_tags WHERE sequence_id = ?", (sequence.id,))
        for tag in sequence.tags:
            cursor.execute(
                "INSERT INTO technique_tags (tag, sequence_id) VALUES (?, ?)",
                (tag, sequence.id)
            )
            
        # Update command stats
        for cmd_entry in sequence.commands:
            cmd = cmd_entry.get("command", "")
            success = cmd_entry.get("success", False) 
            importance = cmd_entry.get("importance", 1)
            
            # Get current stats
            cursor.execute(
                "SELECT success_count, usage_count, avg_importance FROM command_stats WHERE command = ?",
                (cmd,)
            )
            row = cursor.fetchone()
            
            if row:
                success_count, usage_count, avg_importance = row
                
                # Update stats
                new_usage_count = usage_count + 1
                new_success_count = success_count + (1 if success else 0)
                new_avg_importance = ((avg_importance * usage_count) + importance) / new_usage_count
                
                cursor.execute(
                    """
                    UPDATE command_stats 
                    SET success_count = ?, usage_count = ?, avg_importance = ?
                    WHERE command = ?
                    """,
                    (new_success_count, new_usage_count, new_avg_importance, cmd)
                )
            else:
                # Insert new stats
                cursor.execute(
                    """
                    INSERT INTO command_stats 
                    (command, success_count, usage_count, avg_importance)
                    VALUES (?, ?, ?, ?)
                    """,
                    (cmd, 1 if success else 0, 1, importance)
                )
        
        conn.commit()
        conn.close()
        
    def get_command_sequences(self, 
                            target_type: Optional[str] = None, 
                            tags: Optional[List[str]] = None,
                            successful_only: bool = False,
                            with_flags: bool = False,
                            limit: int = 10) -> List[CommandSequence]:
        """Get command sequences matching the criteria"""
        # Start with all sequences
        sequence_ids = set(seq.id for seq in self.command_sequences)
        
        # Filter by target type
        if target_type and target_type in self.target_type_index:
            sequence_ids &= self.target_type_index[target_type]
            
        # Filter by tags (any tag matches)
        if tags:
            tag_matches = set()
            for tag in tags:
                if tag in self.technique_tags:
                    tag_matches.update(self.technique_tags[tag])
            if tag_matches:
                sequence_ids &= tag_matches
                
        # Get the actual sequences
        sequences = []
        for seq in self.command_sequences:
            if seq.id in sequence_ids:
                # Filter by success
                if successful_only and seq.success_level < 2:
                    continue
                    
                # Filter by flags
                if with_flags and not seq.flag_found:
                    continue
                    
                sequences.append(seq)
                
        # Sort by success level and recency
        sequences.sort(key=lambda s: (s.success_level, s.created_at), reverse=True)
        
        return sequences[:limit]
        
    def _extract_command_pattern(self, command: str) -> str:
        """Extract a pattern from a command for similarity matching"""
        # Remove specific parameters to generalize the pattern
        # Example: "nmap -p 80,443 192.168.1.1" -> "nmap -p NUM IP"
        
        # Extract the base command
        parts = command.split(None, 1)
        if not parts:
            return ""
            
        base_cmd = parts[0].lower()
        
        # Apply command-specific patterns
        if base_cmd == "nmap":
            return "nmap_scan"
        elif base_cmd in ["ssh", "ftp", "telnet", "nc", "netcat"]:
            return f"{base_cmd}_connect"
        elif base_cmd in ["curl", "wget"]:
            return "http_request"
        elif base_cmd in ["gobuster", "dirb", "ffuf"]:
            return "directory_brute_force"
        elif base_cmd in ["hydra", "john", "hashcat"]:
            return "credential_brute_force"
        elif base_cmd in ["cat", "less", "more", "grep", "find"]:
            return "file_inspection"
        elif base_cmd in ["ls", "dir"]:
            return "directory_listing"
        elif base_cmd in ["python", "python3", "perl", "php", "ruby"]:
            return "script_execution"
        elif base_cmd in ["gcc", "g++", "make", "javac"]:
            return "compilation"
        elif base_cmd in ["msfconsole", "msfvenom"]:
            return "metasploit"
        else:
            # Generic pattern for other commands
            return base_cmd
            
    def get_similar_commands(self, command: str, limit: int = 5) -> List[str]:
        """Get similar commands that have been successful in the past"""
        pattern = self._extract_command_pattern(command)
        
        if not pattern or pattern not in self.command_pattern_index:
            return []
            
        # Get sequences matching this pattern
        sequence_ids = self.command_pattern_index[pattern]
        
        # Extract successful commands from these sequences
        successful_commands = []
        for seq in self.command_sequences:
            if seq.id in sequence_ids and seq.success_level >= 2:
                for cmd_entry in seq.commands:
                    if cmd_entry.get("success", False):
                        cmd = cmd_entry.get("command", "")
                        if cmd and cmd != command:  # Don't include the exact same command
                            # Score based on importance and sequence success
                            score = cmd_entry.get("importance", 1) * seq.success_level
                            successful_commands.append((cmd, score))
                            
        # Sort by score and return the top commands
        successful_commands.sort(key=lambda x: x[1], reverse=True)
        return [cmd for cmd, _ in successful_commands[:limit]]
        
    def get_next_recommended_commands(self, 
                                    last_command: str,
                                    target_type: str = "unknown",
                                    limit: int = 3) -> List[str]:
        """Get commands recommended to run after the given command"""
        pattern = self._extract_command_pattern(last_command)
        
        if not pattern:
            return []
            
        # Find sequences where this command pattern was used
        matching_sequences = []
        for seq in self.command_sequences:
            if seq.target_type == target_type and seq.success_level >= 2:
                # Check if the sequence contains a command matching this pattern
                for i, cmd_entry in enumerate(seq.commands):
                    cmd = cmd_entry.get("command", "")
                    if self._extract_command_pattern(cmd) == pattern:
                        # Found a match - look for the next command
                        if i < len(seq.commands) - 1:
                            next_cmd = seq.commands[i + 1].get("command", "")
                            if next_cmd:
                                score = seq.success_level
                                matching_sequences.append((next_cmd, score))
                                break
                                
        # Sort by score and remove duplicates
        matching_sequences.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        recommended = []
        
        for cmd, _ in matching_sequences:
            if cmd not in seen:
                seen.add(cmd)
                recommended.append(cmd)
                if len(recommended) >= limit:
                    break
                    
        return recommended
        
    def get_flag_finding_sequences(self, limit: int = 3) -> List[CommandSequence]:
        """Get sequences that successfully found flags"""
        flag_sequences = [seq for seq in self.command_sequences if seq.flag_found]
        # Sort by recency
        flag_sequences.sort(key=lambda s: s.created_at, reverse=True)
        return flag_sequences[:limit]
        
    def extract_technique_description(self, sequence: CommandSequence) -> str:
        """Extract a concise description of the technique used in a sequence"""
        if sequence.description:
            return sequence.description
            
        # Generate a description based on the sequence
        cmd_types = set()
        for cmd_entry in sequence.commands:
            cmd = cmd_entry.get("command", "")
            pattern = self._extract_command_pattern(cmd)
            if pattern:
                cmd_types.add(pattern)
                
        if sequence.flag_found:
            target_desc = f"on {sequence.target_type} target" if sequence.target_type != "unknown" else ""
            techniques = ", ".join(cmd_types)
            return f"Successful flag capture {target_desc} using {techniques}"
        elif sequence.success_level >= 2:
            return f"Partially successful technique on {sequence.target_type} using {', '.join(cmd_types)}"
        else:
            return f"Command sequence for {sequence.target_type} exploration"


class LearningSystem:
    """System for tracking and applying learned command sequences"""
    
    def __init__(self, persistence_path: str = "./learning_data"):
        self.persistence_path = persistence_path
        self.technique_library = TechniqueLibrary(persistence_path=persistence_path)
        
        # Current active command sequence
        self.current_sequence = None
        
        # Target information
        self.target_type = "unknown"
        self.target_tags = []
        
        # Flag detection patterns
        self.flag_patterns = [
            r'flag\{[\w\d_\-\+\*\(\)]+\}',
            r'htb\{[\w\d_\-\+\*\(\)]+\}',
            r'thm\{[\w\d_\-\+\*\(\)]+\}',
            r'root:[\w\d_\-\+\*\(\)]+',
            r'user:[\w\d_\-\+\*\(\)]+',
            r'flag\s*:\s*[\w\d_\-\+\*\(\)]+',
            r'key\s*:\s*[\w\d_\-\+\*\(\)]+',
            r'password\s*:\s*[\w\d_\-\+\*\(\)]+',
            r'the flag is\s*:?\s*[\w\d_\-\+\*\(\)]+',
        ]
        
    def start_tracking(self, description: str = "", target_type: str = "unknown", tags: List[str] = None):
        """Start tracking a new command sequence"""
        self.current_sequence = CommandSequence(
            name=f"Session {time.strftime('%Y-%m-%d %H:%M')}",
            description=description,
            target_type=target_type,
            tags=tags or []
        )
        self.target_type = target_type
        self.target_tags = tags or []
        print(f"[Learning] Started tracking new command sequence: {self.current_sequence.id}")
        
    def track_command(self, command: str, result: str, detect_success: bool = True):
        """Track a command execution and its result"""
        if not self.current_sequence:
            self.start_tracking()
            
        # Detect if this command was successful
        success = False
        importance = 1
        
        if detect_success:
            # Check for indicators of success in the output
            success_patterns = [
                r'success',
                r'found',
                r'vulnerab(le|ility)',
                r'exploit',
                r'shell',
                r'root',
                r'admin',
                r'login successful',
                r'authenticated',
                r'opened connection',
                r'permissions',
                r'password'
            ]
            
            for pattern in success_patterns:
                if re.search(pattern, result, re.IGNORECASE):
                    success = True
                    break
                    
            # Check for flag patterns
            extracted_flag = None
            for pattern in self.flag_patterns:
                matches = re.search(pattern, result, re.IGNORECASE)
                if matches:
                    extracted_flag = matches.group(0)
                    success = True
                    importance = 5  # Flag-finding commands are very important
                    
                    # Mark the sequence as having found a flag
                    self.current_sequence.mark_flag_found(extracted_flag)
                    print(f"[Learning] Flag detected: {extracted_flag}")
                    break
        
        # Add the command to the sequence
        self.current_sequence.add_command(
            command=command,
            result=result,
            success=success,
            importance=importance
        )
        
        # If this is the first command, try to determine target type
        if len(self.current_sequence.commands) == 1:
            if "nmap" in command.lower():
                self.current_sequence.target_type = "network"
                self.target_type = "network"
            elif any(web_term in command.lower() for web_term in ["http", "curl", "wget", "web"]):
                self.current_sequence.target_type = "web"
                self.target_type = "web"
            elif any(term in command.lower() for term in ["ssh", "linux", "bash"]):
                self.current_sequence.target_type = "linux"
                self.target_type = "linux"
            elif any(term in command.lower() for term in ["windows", "powershell", "cmd.exe"]):
                self.current_sequence.target_type = "windows"
                self.target_type = "windows"
        
        print(f"[Learning] Tracked command: {command[:50]}... success={success}")
        return success
        
    def complete_tracking(self, success_level: int = 0, outcome: str = ""):
        """Complete tracking the current sequence"""
        if not self.current_sequence:
            return
            
        # Determine overall success level if not specified
        if success_level == 0:
            # Count successful commands
            success_count = sum(1 for cmd in self.current_sequence.commands if cmd.get("success", False))
            
            if self.current_sequence.flag_found:
                success_level = 3  # Complete success if flag found
            elif success_count > len(self.current_sequence.commands) // 2:
                success_level = 2  # Partial success if majority of commands successful
            elif success_count > 0:
                success_level = 1  # Minimal success if at least one command was successful
                
        # Set the success level and outcome
        self.current_sequence.mark_success(success_level, outcome)
        
        # Save to the library
        self.technique_library.add_command_sequence(self.current_sequence)
        
        print(f"[Learning] Completed tracking sequence {self.current_sequence.id} with success level {success_level}")
        
        # Reset current sequence
        completed_sequence = self.current_sequence
        self.current_sequence = None
        
        return completed_sequence
        
    def get_recommendations(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get recommendations based on learning data and current context"""
        if not context:
            context = {}
            
        # Get target type from context or use default
        target_type = context.get("target_type", self.target_type)
        
        # Get last command if available
        last_command = ""
        if self.current_sequence and self.current_sequence.commands:
            last_command = self.current_sequence.commands[-1].get("command", "")
        
        # Create recommendations structure
        recommendations = {
            "similar_commands": [],
            "next_steps": [],
            "successful_techniques": [],
            "flag_techniques": []
        }
        
        # Get similar commands
        if last_command:
            recommendations["similar_commands"] = self.technique_library.get_similar_commands(
                last_command, limit=3
            )
            
            # Get recommended next commands
            recommendations["next_steps"] = self.technique_library.get_next_recommended_commands(
                last_command, target_type, limit=3
            )
            
        # Get successful techniques for this target type
        successful_sequences = self.technique_library.get_command_sequences(
            target_type=target_type,
            successful_only=True,
            limit=2
        )
        
        for seq in successful_sequences:
            technique_desc = self.technique_library.extract_technique_description(seq)
            key_commands = [cmd.get("command") for cmd in seq.commands if cmd.get("importance", 0) >= 2][:3]
            
            recommendations["successful_techniques"].append({
                "description": technique_desc,
                "key_commands": key_commands
            })
            
        # Get flag-finding techniques
        flag_sequences = self.technique_library.get_flag_finding_sequences(limit=2)
        for seq in flag_sequences:
            technique_desc = self.technique_library.extract_technique_description(seq)
            key_commands = [cmd.get("command") for cmd in seq.commands if cmd.get("success", False)][:3]
            
            recommendations["flag_techniques"].append({
                "description": technique_desc,
                "target_type": seq.target_type,
                "key_commands": key_commands
            })
            
        return recommendations


def init_learning_system(persistence_path: str = "./learning_data") -> LearningSystem:
    """Initialize the learning system"""
    return LearningSystem(persistence_path=persistence_path)


def integrate_learning_with_agent(agent, learning_system: LearningSystem):
    """Integrate learning system with DarkCircuit agent"""
    # Store reference to learning system
    agent.learning_system = learning_system
    
    # If the agent has a run_command function, wrap it to track commands
    original_run_command = getattr(agent, 'run_command', None)
    
    if original_run_command:
        async def learning_run_command(command: str) -> str:
            # Execute the command
            result = await original_run_command(command)
            
            # Track the command in the learning system
            learning_system.track_command(command, result)
            
            return result
            
        # Replace the agent's run_command
        agent.run_command = learning_run_command
        
    # Add a function to get learning recommendations
    def get_learning_recommendations(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return learning_system.get_recommendations(context)
        
    agent.get_learning_recommendations = get_learning_recommendations
    
    # Modify the agent's run_agent_streaming to start and complete tracking
    original_run_agent = getattr(agent, 'run_agent_streaming', None)
    
    if original_run_agent:
        async def learning_run_agent(prompt: str):
            # Start tracking a new command sequence
            target_type = "unknown"
            # Try to determine target type from prompt
            if any(term in prompt.lower() for term in ["web", "http", "website", "url"]):
                target_type = "web"
            elif any(term in prompt.lower() for term in ["linux", "bash", "ssh"]):
                target_type = "linux"
            elif any(term in prompt.lower() for term in ["windows", "powershell", "cmd"]):
                target_type = "windows"
                
            tags = []
            # Extract potential tags from prompt
            if "hack the box" in prompt.lower() or "htb" in prompt.lower():
                tags.append("hackthebox")
            if "tryhackme" in prompt.lower() or "thm" in prompt.lower():
                tags.append("tryhackme")
            if "ctf" in prompt.lower():
                tags.append("ctf")
                
            learning_system.start_tracking(
                description=f"Session from prompt: {prompt[:50]}...",
                target_type=target_type,
                tags=tags
            )
            
            try:
                # Execute the original function
                async for event in original_run_agent(prompt):
                    # Check if this is the final event
                    if event.get("type") == "token" and "[Ready to answer]" in event.get("value", ""):
                        # Mark as complete with outcome from response
                        learning_system.complete_tracking(
                            success_level=2 if learning_system.current_sequence.flag_found else 1,
                            outcome=event.get("value", "")
                        )
                        
                    yield event
            finally:
                # Always complete tracking even on error
                if learning_system.current_sequence:
                    learning_system.complete_tracking()
                    
        # Replace the agent's run function
        agent.run_agent_streaming = learning_run_agent