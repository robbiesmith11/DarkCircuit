"""
Tools for the DarkCircuit Agent.

This module provides the tool definitions for the DarkCircuit Agent,
including command execution and RAG retrieval.
"""

from langchain_core.tools import tool
from typing import Callable, Awaitable, Dict, Any, Optional

# Import the RAG tool if available, otherwise create a placeholder function
try:
    from Rag_tool import load_static_rag_context
except ImportError:
    def load_static_rag_context():
        print("RAG tool not available. Create Rag_tool.py with load_static_rag_context() function to enable this feature.")
        return None


class AgentTools:
    """
    Tools for the DarkCircuit Agent.

    This class encapsulates the tools used by the agent, including:
    - run_command: Execute a command on the SSH server
    - rag_retrieve: Search for relevant security documents
    """

    def __init__(self,
                 streaming_handler=None,
                 ssh_command_runner: Optional[Callable[[str, int], Awaitable[Dict[str, Any]]]] = None):
        """
        Initialize the agent tools.

        Args:
            streaming_handler: The streaming handler for sending events
            ssh_command_runner: Optional function to execute SSH commands directly
        """
        self.streaming_handler = streaming_handler
        self.ssh_command_runner = ssh_command_runner
        self.terminal_command_id = 0
        self.terminal_output_queue = None

        # Initialize async queue when needed
        if not self.ssh_command_runner:
            import asyncio
            self.terminal_output_queue = asyncio.Queue()

    @tool
    async def run_command(self, command: str) -> str:
        """Execute a command on the remote SSH server."""
        from agent_utils import optimize_command, wait_for_terminal_output

        try:
            # Generate a unique ID for this command execution
            self.terminal_command_id += 1
            command_id = self.terminal_command_id

            # Optimize the command if possible
            optimized_command, optimization_message = optimize_command(command)

            # If we have a direct SSH command runner, use it
            if self.ssh_command_runner:
                print(f"Executing command directly on SSH: {optimized_command} (ID: {command_id})")

                # Still notify the frontend about the command for display purposes
                if self.streaming_handler:
                    await self.streaming_handler.queue.put({
                        "type": "ui_terminal_command",
                        "command": optimized_command,
                        "command_id": command_id
                    })

                # Execute the command directly using the provided function
                result = await self.ssh_command_runner(optimized_command)

                # Extract the output
                if result and "success" in result:
                    output = result.get("output", "")
                    error = result.get("error", "")

                    # Combine output and error for proper context
                    full_output = output
                    if error:
                        full_output += f"\n\nErrors:\n{error}"

                    # If there was an optimization, prepend that information
                    if optimization_message:
                        full_output = f"Note: {optimization_message}\n\n{full_output}"

                    return full_output
                else:
                    return "Command execution failed or returned no results."
            else:
                # Fallback to the old method using the frontend
                print(f"No direct SSH runner available, using frontend for command: {optimized_command} (ID: {command_id})")

                # Tell the frontend to execute the command
                if self.streaming_handler:
                    await self.streaming_handler.queue.put({
                        "type": "ui_terminal_command",
                        "command": optimized_command,
                        "command_id": command_id
                    })

                # Wait for output from the frontend
                if self.terminal_output_queue:
                    return await wait_for_terminal_output(self.terminal_output_queue, command_id)
                else:
                    return "Terminal output queue not initialized."

        except Exception as e:
            return f"Error executing command: {str(e)}"

    @tool
    def rag_retrieve(self, query: str) -> str:
        """Search for relevant documents included are writeups from hack the box challenges using RAG"""
        retriever = load_static_rag_context()
        if not retriever:
            return "RAG functionality is not available. Please check the Rag_tool.py implementation."

        docs = retriever.get_relevant_documents(query)
        content_parts = []
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            content_parts.append(f"[Source {i + 1}] {doc.page_content}")
        return "\n\n".join(content_parts)
