"""
StreamingHandler module for DarkCircuit Agent.

This module provides a streaming callback handler for LangChain 
to handle streaming tokens, tool calls, and thinking process.
"""

import asyncio
from langchain_core.callbacks.base import BaseCallbackHandler


class StreamingHandler(BaseCallbackHandler):
    """
    Custom streaming handler for LangChain to process tokens and tool events.
    
    This handler manages different types of events:
    - Token events (actual output to show users)
    - Thinking events (internal reasoning process)
    - Tool calls and results (commands and their outputs)
    
    All events are put into an asyncio queue for streaming to clients.
    """
    
    def __init__(self, output_target="chat"):
        """Initialize the streaming handler.
        
        Args:
            output_target (str): Either "chat" or "debug" to determine how tokens are handled
        """
        self.queue = asyncio.Queue()
        self.output_target = output_target
        self.buffer = ""
        self.thinking_buffer = ""  # Buffer specifically for thinking tokens
        self.current_phase = None  # Track what phase we're in (reasoning/response)

    async def on_llm_new_token(self, token: str, **kwargs):
        """Handle new tokens from the LLM.
        
        Args:
            token (str): The token from the LLM
            **kwargs: Additional arguments
        """
        # Determine if this is a thinking token or chat token based on output_target
        event_type = "token" if self.output_target == "chat" else "thinking"
        print(f"[StreamingHandler] {event_type}: {token}")

        if event_type == "thinking":
            # For thinking tokens, accumulate in the buffer instead of sending immediately
            self.thinking_buffer += token
        else:
            # For chat tokens, send immediately
            await self.queue.put({"type": "token", "value": token})

    async def on_tool_start(self, tool, input_str, **kwargs):
        """Handle tool start events.
        
        Args:
            tool: The tool being called
            input_str: The input to the tool
            **kwargs: Additional arguments
        """
        # When a tool starts, we should flush any thinking buffer
        await self._flush_thinking_buffer()

        # Tool is already a dictionary, so we can access it directly
        tool_name = tool.get('name', 'unknown_tool')
        tool_description = tool.get('description', 'unknown')

        # Convert input_str from Python dict-like string to actual dict
        command = input_str
        try:
            # This is a safer approach to convert a Python dict string to a dict
            import ast
            input_dict = ast.literal_eval(input_str)
            if isinstance(input_dict, dict) and 'command' in input_dict:
                command = input_dict['command']
        except:
            # If parsing fails, keep the original string
            pass

        print(f"[StreamingHandler] Starting tool: {tool_name} with input: {command}")

        # Check if this is the context-aware RAG tool
        is_context_aware = tool_name == "rag_retrieve_with_context"
        
        # Standardize the format for tool calls
        await self.queue.put({
            "type": "tool_call",
            "name": tool_name,
            "description": tool_description,
            "input": command,
            "is_context_aware": is_context_aware
        })

    async def on_tool_end(self, output, **kwargs):
        """Handle tool end events.
        
        Args:
            output: The output from the tool
            **kwargs: Additional arguments
        """
        print(f"[StreamingHandler] Tool result: {output}")

        result_content = ""
        # Extract content from different possible formats
        if hasattr(output, 'content'):
            result_content = output.content
        elif isinstance(output, dict):
            result_content = output.get("content", str(output))
        else:
            result_content = str(output)

        # Standardize the format for tool results
        await self.queue.put({
            "type": "tool_result",
            "output": result_content
        })

    async def _flush_thinking_buffer(self, done=False):
        """Flush accumulated thinking content to the queue"""
        if self.thinking_buffer.strip():
            # If done, append image HTML or markdown to the content
            value = self.thinking_buffer.strip()
            if done:
                value += "\n\n![](http://127.0.0.1:8000/GAMEX_LOGO.png)"  # or /assets/... if local

            await self.queue.put({
                "type": "thinking",
                "value": value,
                "done": done
            })
            self.thinking_buffer = ""  # Clear the buffer after sending

    async def stream(self):
        """Stream events from the queue.
        
        Yields:
            dict: Event objects from the queue
        """
        while True:
            item = await self.queue.get()
            if item == "__END__":
                break
            yield item

    async def end(self):
        """End the streaming session and clean up."""
        # Make sure to flush any remaining thinking content
        await self._flush_thinking_buffer()

        if self.buffer.strip():
            await self.queue.put({"type": "token", "value": self.buffer})
        await self.queue.put("__END__")
