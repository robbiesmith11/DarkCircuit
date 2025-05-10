"""
Utility functions for the DarkCircuit Agent.

This module provides various utility functions
used by the agent for command execution and state management.
"""

import os
import asyncio
import json
import time
from typing import Dict, Any, Optional


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


def optimize_command(command: str) -> (str, Optional[str]):
    """
    Optimize potentially slow commands.
    
    Args:
        command (str): The command to optimize
        
    Returns:
        tuple: (optimized_command, optimization_message)
    """
    optimization_message = None
    
    # Optimize nmap commands
    if 'nmap' in command and '--min-rate' not in command:
        # Check if this is a comprehensive scan that will take a long time
        if any(flag in command for flag in ['-sS', '-sV', '-A', '-p-']):
            # Add min-rate parameter if not already present
            optimized_command = command + ' --min-rate=1000' if '--min-rate' not in command else command
            optimization_message = f"Optimizing potentially slow nmap command with --min-rate=1000 to speed up execution."
            print(f"[Agent] Optimized slow nmap command: {command} -> {optimized_command}")
            return optimized_command, optimization_message
    
    # Add more command optimizations here as needed
    
    return command, optimization_message
