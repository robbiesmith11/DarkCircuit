"""
Utility functions for the DarkCircuit Agent.

This module provides various utility functions
used by the agent for command execution and state management.
"""

import os
import asyncio
import json
import time
import re
from typing import Dict, Any, Optional, Tuple


def load_prompts():
    """
    Load prompts from the shared JSON file in the frontend public directory.
    
    Returns:
        tuple: A tuple containing (reasoner_prompt, responder_prompt)
    """
    try:
        # Try multiple locations for the prompts.json file
        possible_paths = [
            'prompts.json',
            'frontend/public/prompts.json',
            'frontend/dist/prompts.json'
        ]
        
        for prompt_file_path in possible_paths:
            if os.path.exists(prompt_file_path):
                with open(prompt_file_path, 'r') as file:
                    prompts = json.load(file)
                    print(f"Loaded prompts from {prompt_file_path}")
                    return prompts.get('reasonerPrompt'), prompts.get('responderPrompt')
        
        # If we can't find the file, use hardcoded default prompts
        print("Could not find prompts.json. Using default prompts.")
        
        DEFAULT_REASONER_PROMPT = """You are a computer security expert specialized in penetration testing and security research.
You help users understand security concepts, perform security assessments, and learn how to identify and exploit vulnerabilities in systems.
The user has access to a terminal through which they can run commands on a target system. You should guide them through the process of exploring the system, understanding its security posture, and identifying potential vulnerabilities.

Your task is to reason step by step about the security challenges presented by the user. Use the run_command tool to execute commands on the target system.
Use the rag_retrieve tool to search for relevant documents about similar security challenges.

When you've thoroughly analyzed the situation and are ready to provide a final answer, include the phrase [Ready to answer] in your response."""
        
        DEFAULT_RESPONDER_PROMPT = """You are a computer security expert specialized in penetration testing and security research.
You help users understand security concepts, perform security assessments, and learn how to identify and exploit vulnerabilities in systems.
The user has access to a terminal through which they can run commands on a target system. You should guide them through the process of exploring the system, understanding its security posture, and identifying potential vulnerabilities.

Based on the reasoning and analysis performed, provide a clear, concise and helpful response to the user. Include any relevant commands they should run, explanations of security concepts, and step-by-step guidance as needed.

Remember to be ethical and professional in your guidance. Focus on education and legitimate security purposes."""
        
        return DEFAULT_REASONER_PROMPT, DEFAULT_RESPONDER_PROMPT
        
    except Exception as e:
        print(f"Error loading prompts: {str(e)}")
        return None, None


async def wait_for_terminal_output(queue, command_id, timeout=1200):
    """
    Wait for terminal output from the frontend.
    
    Args:
        queue: The asyncio.Queue to receive terminal output
        command_id (int): The ID of the command to wait for
        timeout (int): Maximum time to wait in seconds
        
    Returns:
        str: The terminal output or a timeout message
    """
    print(f"Waiting for terminal output from frontend for command {command_id}")
    
    # Simple implementation that waits for up to the specified timeout
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # Check if we have data in the queue
        try:
            # Use a small timeout to avoid blocking forever
            data = await asyncio.wait_for(queue.get(), timeout=1.0)
            
            if data and isinstance(data, dict):
                # Check if this is for our command ID
                if data.get("command_id") == command_id:
                    output = data.get("output", "")
                    return output
        except asyncio.TimeoutError:
            # Just keep waiting
            await asyncio.sleep(0.5)
    
    # If we get here, we timed out
    return f"Command timed out after {timeout} seconds"


def sanitize_command(command: str) -> Tuple[str, Optional[str]]:
    """
    Sanitize commands to prevent targeting localhost or the user's own system.
    Also provide context awareness for HTB environment.
    
    Args:
        command (str): The command to sanitize
        
    Returns:
        tuple: (sanitized_command, sanitization_message)
    """
    sanitization_message = None
    original_command = command
    
    # Patterns to identify localhost targeting
    localhost_patterns = [
        r'(^|\s)(localhost)(\s|$|:|/)',
        r'(^|\s)(127\.0\.0\.1)(\s|$|:|/)',
        r'(^|\s)(::1)(\s|$|:|/)'
    ]
    
    # Patterns that indicate local username confusion (common issue)
    local_username_patterns = [
        r'(^|\s)(/home/[^/\s]+)(\s|$)',  # Local home directories
        r'(^|\s)(~/[^\s]*)(\s|$)',       # Local home shortcuts
    ]
    
    # Commands that should not target localhost
    risky_command_patterns = [
        r'curl\s+',
        r'wget\s+',
        r'nmap\s+',
        r'hydra\s+.*ssh',
        r'nikto\s+',
        r'sqlmap\s+',
        r'dirb\s+',
        r'gobuster\s+',
        r'wpscan\s+',
        r'ssh\s+',
        r'nc\s+',
        r'netcat\s+',
        r'ftp\s+',
        r'telnet\s+',
        r'smbclient\s+'
    ]
    
    # Check if this is a risky command
    is_risky_command = any(re.search(pattern, command) for pattern in risky_command_patterns)
    
    if is_risky_command:
        # Check if targeting localhost
        targets_localhost = any(re.search(pattern, command) for pattern in localhost_patterns)
        
        if targets_localhost:
            # Replace localhost references with a placeholder to avoid targeting local system
            sanitized_command = command
            for pattern in localhost_patterns:
                sanitized_command = re.sub(pattern, r'\1$TARGET_IP\3', sanitized_command)
            
            sanitization_message = f"SECURITY: Command modified to use $TARGET_IP instead of localhost. You are on HTB Pwnbox - target the challenge machine, not localhost!"
            print(f"[Agent] Sanitized localhost command: {command} -> {sanitized_command}")
            return sanitized_command, sanitization_message
    
    # Check for local path confusion in file operations
    has_local_paths = any(re.search(pattern, command) for pattern in local_username_patterns)
    if has_local_paths and not any(cmd in command for cmd in ['cd', 'ls', 'cat', 'grep', 'find']):
        sanitization_message = f"WARNING: Command contains local paths. Remember you're on HTB Pwnbox - adjust paths for the challenge environment."
        print(f"[Agent] Local path warning for command: {command}")
    
    # If no sanitization was needed, return the original command
    return command, sanitization_message


def optimize_command(command: str) -> Tuple[str, Optional[str]]:
    """
    Optimize potentially slow commands and sanitize them for security.
    
    Args:
        command (str): The command to optimize and sanitize
        
    Returns:
        tuple: (optimized_command, combined_message)
    """
    # First sanitize the command
    sanitized_command, sanitization_message = sanitize_command(command)
    
    # Then optimize it
    optimization_message = None
    
    # Optimize nmap commands
    if 'nmap' in sanitized_command and '--min-rate' not in sanitized_command:
        # Check if this is a comprehensive scan that will take a long time
        if any(flag in sanitized_command for flag in ['-sS', '-sV', '-A', '-p-']):
            # Add min-rate parameter if not already present
            optimized_command = sanitized_command + ' --min-rate=1000' if '--min-rate' not in sanitized_command else sanitized_command
            optimization_message = f"Optimizing potentially slow nmap command with --min-rate=1000 to speed up execution."
            print(f"[Agent] Optimized slow nmap command: {sanitized_command} -> {optimized_command}")
            return optimized_command, combine_messages(sanitization_message, optimization_message)
    
    # Add more command optimizations here as needed
    
    return sanitized_command, sanitization_message


def combine_messages(message1: Optional[str], message2: Optional[str]) -> Optional[str]:
    """Combine two messages into one if both are not None."""
    if message1 and message2:
        return f"{message1} {message2}"
    elif message1:
        return message1
    elif message2:
        return message2
    else:
        return None
