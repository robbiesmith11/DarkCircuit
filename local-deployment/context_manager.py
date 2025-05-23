"""
Intelligent Context Manager for DarkCircuit

Manages conversation context efficiently to balance memory retention 
with token cost optimization for long pentesting sessions.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage


class ContextManager:
    """
    Manages conversation context with intelligent compression and prioritization.
    """
    
    def __init__(self, max_context_tokens: int = 8000, critical_info_tokens: int = 2000):
        """
        Initialize context manager.
        
        Args:
            max_context_tokens: Maximum tokens to maintain in context
            critical_info_tokens: Tokens reserved for critical information
        """
        self.max_context_tokens = max_context_tokens
        self.critical_info_tokens = critical_info_tokens
        self.session_summary = {}
        self.target_info = {}
        self.discovered_vulns = []
        self.key_findings = []
        
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def extract_critical_info(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """
        Extract and categorize critical information from messages.
        
        Returns:
            dict: Categorized critical information
        """
        critical_info = {
            "target_info": {},
            "vulnerabilities": [],
            "credentials": [],
            "network_info": {},
            "key_commands": [],
            "current_objective": None
        }
        
        for msg in messages:
            content = getattr(msg, "content", "")
            
            # Extract target information
            self._extract_target_info(content, critical_info)
            
            # Extract vulnerability findings
            self._extract_vulnerabilities(content, critical_info)
            
            # Extract credentials
            self._extract_credentials(content, critical_info)
            
            # Extract network information
            self._extract_network_info(content, critical_info)
            
            # Extract successful commands
            self._extract_key_commands(content, critical_info)
            
        return critical_info
    
    def _extract_target_info(self, content: str, critical_info: Dict):
        """Extract target system information"""
        patterns = {
            "ip_addresses": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            "domains": r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b',
            "ports": r'port (\d+)',
            "services": r'(ssh|http|https|ftp|smtp|mysql|postgresql|smb|rdp)',
            "os_info": r'(Windows|Linux|Ubuntu|CentOS|Red Hat|Debian)',
        }
        
        for category, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                if category not in critical_info["target_info"]:
                    critical_info["target_info"][category] = set()
                critical_info["target_info"][category].update(matches)
    
    def _extract_vulnerabilities(self, content: str, critical_info: Dict):
        """Extract vulnerability information"""
        vuln_patterns = [
            r'CVE-\d{4}-\d{4,}',
            r'SQL injection',
            r'XSS|Cross-site scripting',
            r'Remote code execution',
            r'Buffer overflow',
            r'Authentication bypass',
            r'Privilege escalation',
            r'Directory traversal',
            r'CSRF|Cross-site request forgery'
        ]
        
        for pattern in vuln_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                vuln_info = {
                    "type": pattern,
                    "context": content[:200] + "..." if len(content) > 200 else content,
                    "timestamp": datetime.now().isoformat()
                }
                critical_info["vulnerabilities"].append(vuln_info)
    
    def _extract_credentials(self, content: str, critical_info: Dict):
        """Extract credential information"""
        cred_patterns = [
            r'username[:\s]+([^\s\n]+)',
            r'password[:\s]+([^\s\n]+)',
            r'hash[:\s]+([a-fA-F0-9]{32,})',
            r'token[:\s]+([a-zA-Z0-9\-_\.]{20,})',
        ]
        
        for pattern in cred_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match) > 3:  # Filter out very short matches
                    critical_info["credentials"].append({
                        "type": pattern.split('[')[0],
                        "value": match,
                        "context": content[:100] + "..."
                    })
    
    def _extract_network_info(self, content: str, critical_info: Dict):
        """Extract network topology information"""
        network_patterns = {
            "subnets": r'\b(?:\d{1,3}\.){3}\d{1,3}/\d{1,2}\b',
            "mac_addresses": r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b',
            "hostnames": r'hostname[:\s]+([^\s\n]+)',
        }
        
        for category, pattern in network_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                if category not in critical_info["network_info"]:
                    critical_info["network_info"][category] = set()
                critical_info["network_info"][category].update(matches)
    
    def _extract_key_commands(self, content: str, critical_info: Dict):
        """Extract successful/important commands"""
        if "successfully" in content.lower() or "found" in content.lower():
            # Extract command from tool messages
            if "run_command" in content:
                command_match = re.search(r'run_command.*?([^\n]+)', content)
                if command_match:
                    critical_info["key_commands"].append({
                        "command": command_match.group(1).strip(),
                        "result_summary": content[:150] + "...",
                        "timestamp": datetime.now().isoformat()
                    })
    
    def compress_tool_results(self, content: str, tool_name: str) -> str:
        """
        Compress verbose tool results while preserving key information.
        
        Args:
            content: Original tool result content
            tool_name: Name of the tool that generated the result
            
        Returns:
            str: Compressed version of the content
        """
        if tool_name == "run_command":
            return self._compress_command_output(content)
        elif tool_name == "rag_retrieve_tool":
            return self._compress_rag_output(content)
        else:
            return self._generic_compression(content)
    
    def _compress_command_output(self, content: str) -> str:
        """Compress command output intelligently"""
        # For nmap scans
        if "nmap" in content.lower():
            return self._compress_nmap_output(content)
        
        # For directory listings
        if any(indicator in content for indicator in ["drwx", "total", "-rw-"]):
            return self._compress_directory_listing(content)
        
        # For file contents
        if len(content) > 1000:
            return self._compress_file_content(content)
        
        return content
    
    def _compress_nmap_output(self, content: str) -> str:
        """Compress nmap scan results"""
        lines = content.split('\n')
        compressed = []
        
        for line in lines:
            # Keep open ports
            if "/tcp" in line and "open" in line:
                compressed.append(line.strip())
            # Keep host info
            elif "Nmap scan report" in line:
                compressed.append(line.strip())
            # Keep OS detection
            elif "OS:" in line or "Running:" in line:
                compressed.append(line.strip())
        
        if compressed:
            return "NMAP Results:\n" + "\n".join(compressed)
        return content[:300] + "..." if len(content) > 300 else content
    
    def _compress_directory_listing(self, content: str) -> str:
        """Compress directory listings"""
        lines = content.split('\n')
        files = []
        dirs = []
        
        for line in lines:
            if line.startswith('d'):  # Directory
                dirs.append(line.split()[-1] if line.split() else line)
            elif line.startswith('-'):  # File
                files.append(line.split()[-1] if line.split() else line)
        
        summary = []
        if dirs:
            summary.append(f"Directories: {', '.join(dirs[:10])}")
            if len(dirs) > 10:
                summary.append(f"... and {len(dirs) - 10} more directories")
        
        if files:
            summary.append(f"Files: {', '.join(files[:10])}")
            if len(files) > 10:
                summary.append(f"... and {len(files) - 10} more files")
        
        return "\n".join(summary) if summary else content
    
    def _compress_file_content(self, content: str) -> str:
        """Compress large file contents"""
        lines = content.split('\n')
        
        # For config files, keep important lines
        important_patterns = [
            r'password', r'key', r'secret', r'token', r'admin', r'root',
            r'database', r'connection', r'server', r'host', r'port'
        ]
        
        important_lines = []
        for line in lines:
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in important_patterns):
                important_lines.append(line.strip())
        
        if important_lines:
            result = f"Key lines from file:\n" + "\n".join(important_lines[:20])
            if len(lines) > 50:
                result += f"\n... (file has {len(lines)} total lines)"
            return result
        
        # Fallback: first and last few lines
        if len(lines) > 20:
            return "\n".join(lines[:10]) + "\n...\n" + "\n".join(lines[-10:])
        
        return content
    
    def _compress_rag_output(self, content: str) -> str:
        """Compress RAG tool output"""
        # Keep source citations but summarize content
        sources = re.findall(r'\[Source \d+\]', content)
        
        # Extract key techniques/methods mentioned
        techniques = re.findall(r'(exploit|vulnerability|payload|technique|method|attack)', content, re.IGNORECASE)
        
        summary = f"RAG Sources: {len(sources)} documents"
        if techniques:
            unique_techniques = list(set(techniques[:5]))
            summary += f"\nKey topics: {', '.join(unique_techniques)}"
        
        if len(content) > 500:
            summary += f"\n{content[:200]}..."
        
        return summary
    
    def _generic_compression(self, content: str) -> str:
        """Generic content compression"""
        if len(content) > 500:
            return content[:200] + "\n...\n" + content[-200:]
        return content
    
    def create_context_summary(self, critical_info: Dict[str, Any]) -> str:
        """
        Create a concise summary of the session context.
        
        Args:
            critical_info: Extracted critical information
            
        Returns:
            str: Formatted context summary
        """
        summary_parts = []
        
        # Target information
        if critical_info["target_info"]:
            target_summary = "TARGET INFO:\n"
            for category, items in critical_info["target_info"].items():
                if isinstance(items, set):
                    items = list(items)
                target_summary += f"- {category}: {', '.join(str(i) for i in items[:5])}\n"
            summary_parts.append(target_summary)
        
        # Vulnerabilities
        if critical_info["vulnerabilities"]:
            vuln_summary = f"VULNERABILITIES FOUND: {len(critical_info['vulnerabilities'])}\n"
            for vuln in critical_info["vulnerabilities"][:3]:
                vuln_summary += f"- {vuln['type']}: {vuln['context'][:100]}...\n"
            summary_parts.append(vuln_summary)
        
        # Credentials
        if critical_info["credentials"]:
            cred_summary = f"CREDENTIALS: {len(critical_info['credentials'])} found\n"
            summary_parts.append(cred_summary)
        
        # Key commands
        if critical_info["key_commands"]:
            cmd_summary = "KEY SUCCESSFUL COMMANDS:\n"
            for cmd in critical_info["key_commands"][-5:]:  # Last 5 successful commands
                cmd_summary += f"- {cmd['command']}\n"
            summary_parts.append(cmd_summary)
        
        return "\n".join(summary_parts)
    
    def optimize_context(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Optimize context by compressing messages while preserving critical information.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            List[BaseMessage]: Optimized message list
        """
        # Extract critical information
        critical_info = self.extract_critical_info(messages)
        
        # Calculate current token usage
        total_tokens = sum(self.estimate_tokens(getattr(msg, "content", "")) for msg in messages)
        
        if total_tokens <= self.max_context_tokens:
            return messages  # No optimization needed
        
        print(f"[ContextManager] Optimizing context: {total_tokens} tokens -> target: {self.max_context_tokens}")
        
        # Create context summary
        context_summary = self.create_context_summary(critical_info)
        
        # Keep recent messages and important ones
        optimized_messages = []
        
        # Always keep the original human query
        human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
        if human_messages:
            optimized_messages.append(human_messages[0])  # Original query
        
        # Add context summary as system message
        if context_summary:
            summary_msg = SystemMessage(content=f"SESSION CONTEXT SUMMARY:\n{context_summary}")
            optimized_messages.append(summary_msg)
        
        # Keep recent messages (last N messages based on token budget)
        recent_token_budget = self.max_context_tokens - self.critical_info_tokens
        recent_tokens = 0
        
        for msg in reversed(messages[-20:]):  # Check last 20 messages
            msg_content = getattr(msg, "content", "")
            
            # Compress tool messages
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "unknown")
                compressed_content = self.compress_tool_results(msg_content, tool_name)
                msg_tokens = self.estimate_tokens(compressed_content)
                
                if recent_tokens + msg_tokens <= recent_token_budget:
                    compressed_msg = ToolMessage(
                        content=compressed_content,
                        name=tool_name,
                        tool_call_id=getattr(msg, "tool_call_id", "")
                    )
                    optimized_messages.insert(-1, compressed_msg)  # Insert before summary
                    recent_tokens += msg_tokens
                else:
                    break
            else:
                msg_tokens = self.estimate_tokens(msg_content)
                if recent_tokens + msg_tokens <= recent_token_budget and msg not in optimized_messages:
                    optimized_messages.insert(-1, msg)  # Insert before summary
                    recent_tokens += msg_tokens
        
        final_tokens = sum(self.estimate_tokens(getattr(msg, "content", "")) for msg in optimized_messages)
        print(f"[ContextManager] Context optimized: {len(messages)} -> {len(optimized_messages)} messages, {total_tokens} -> {final_tokens} tokens")
        
        return optimized_messages