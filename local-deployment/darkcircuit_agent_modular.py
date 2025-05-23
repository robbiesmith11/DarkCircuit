"""
DarkCircuit Agent - The main agent implementation.

This module provides the core agent functionality, leveraging LangChain
and LangGraph to create a sophisticated security-focused assistant
that can execute SSH commands and provide security guidance.
"""

import os
import asyncio
from typing import Dict, Any, Callable, Awaitable, Optional, List

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# Import modularized components
from streaming_handler import StreamingHandler

from agent_utils import load_prompts
from Rag_tool import *
from utils import get_path
from context_manager import ContextManager

from dotenv import load_dotenv
load_dotenv(dotenv_path=get_path(".env"))

# Type definitions
MessagesState = dict

class Darkcircuit_Agent:
    """
    The main agent for DarkCircuit security assistant.

    This agent uses LangGraph to implement a reasoning process for security tasks,
    with the ability to execute commands on remote systems and provide analysis.
    """

    def __init__(self,
                 model_name="gpt-4o-mini",
                 reasoning_prompt=None,
                 response_prompt=None,
                 ssh_command_runner: Optional[Callable[[str, int], Awaitable[Dict[str, Any]]]] = None):
        """
        Initialize the agent.

        Args:
            model_name (str): The OpenAI model to use
            reasoning_prompt (str, optional): Custom prompt for the reasoning step
            response_prompt (str, optional): Custom prompt for the response step
            ssh_command_runner (callable, optional): Function to execute SSH commands directly
        """
        # Check for OpenAI API key in environment
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running this script.")

        api_key = os.environ["OPENAI_API_KEY"]

        # Initialize the language model
        self.llm = ChatOpenAI(model=model_name, streaming=True, api_key=api_key)
        self.streaming_handler = None
        self.terminal_command_id = 0

        # Initialize agent tools
        self.ssh_command_runner = ssh_command_runner

        self.search = DuckDuckGoSearchRun()

        @tool
        def rag_retrieve(query: str) -> str:
            """Search for relevant documents included are writeups from hack the box challenges using RAG"""
            retriever = load_static_rag_context()
            docs = retriever.get_relevant_documents(query)
            content_parts = []
            for i, doc in enumerate(docs):
                metadata = doc.metadata
                content_parts.append(f"[Source {i + 1}] {doc.page_content}")
            return "\n\n".join(content_parts)

        @tool
        async def run_command(command: str) -> str:
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

                        # If there was an optimization, prepend that information
                        # if optimization_message:
                        #    output = f"Note: {optimization_message}\n\n{output}"

                        return output
                    else:
                        return "Command execution failed or returned no results."
                else:
                    # Fallback to the old method using the frontend
                    print(
                        f"No direct SSH runner available, using frontend for command: {optimized_command} (ID: {command_id})")

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

        self.run_command = run_command
        self.rag_retrieve = rag_retrieve
        self.tools = [self.search, self.run_command, self.rag_retrieve]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Load default prompts from file or use hardcoded defaults
        DEFAULT_REASONING_PROMPT, DEFAULT_RESPONSE_PROMPT = load_prompts()

        # Create system messages
        self.reasoning_prompt = SystemMessage(content=reasoning_prompt or DEFAULT_REASONING_PROMPT)
        self.response_prompt = SystemMessage(content=response_prompt or DEFAULT_RESPONSE_PROMPT)

        # Initialize context manager for efficient token usage
        self.context_manager = ContextManager(
            max_context_tokens=12000,  # Reasonable limit for GPT-4
            critical_info_tokens=3000   # Reserve tokens for critical info
        )

    def _build_agent_graph(self):
        """Set up the LangGraph for agent reasoning."""
        builder = StateGraph(MessagesState)

        # Add nodes
        builder.add_node("reasoner", self.reasoner)
        builder.add_node("tools", self.tools_node)
        builder.add_node("responder", self.responder)

        # Add edges
        builder.add_edge(START, "reasoner")
        builder.add_conditional_edges("reasoner", self.route_from_reasoner)
        builder.add_edge("tools", "reasoner")
        builder.add_edge("responder", END)

        # Compile the graph
        self.react_graph = builder.compile()

    async def receive_terminal_output(self, output_data):
        """
        Receive terminal output from the frontend.

        Args:
            output_data (dict): Output data from the terminal
        """
        # Find the run_command tool and pass the output to its queue
        for tool in self.tools:
            if hasattr(tool, 'name') and tool.name == 'run_command':
                if hasattr(tool, 'terminal_output_queue'):
                    await tool.terminal_output_queue.put(output_data)
                break

    async def reasoner(self, state: MessagesState):
        """
        Agent reasoning node - determines what actions to take.

        Args:
            state (dict): Current state of the conversation

        Returns:
            dict: Updated state with reasoning results
        """
        print("[Agent] Entering reasoner node with message count:", len(state["messages"]))
        
        # Track step count to prevent infinite loops
        step_count = state.get("step_count", 0) + 1
        max_steps = 35  # Maximum reasoning steps before forcing completion (increased for pentesting scenarios)
        
        print(f"[Agent] Reasoner step {step_count}/{max_steps}")
        
        if step_count >= max_steps:
            print(f"[Agent] Maximum steps ({max_steps}) reached, forcing completion")
            await self.streaming_handler.queue.put({
                "type": "thinking",
                "value": f"Maximum reasoning steps ({max_steps}) reached. Providing final response."
            })
            return {**state, "step_count": step_count, "done": True}

        # Use debug output target for reasoner
        self.streaming_handler.output_target = "debug"

        try:
            # Optimize context for long pentesting sessions
            all_messages = state["messages"]
            optimized_messages = self.context_manager.optimize_context(all_messages)
            
            print(f"[Agent] Context optimization: {len(all_messages)} -> {len(optimized_messages)} messages")
            
            # When preparing messages for the reasoner, include the original user query
            # and system messages, but convert tool messages to system messages
            filtered_messages = []
            for msg in optimized_messages:
                if isinstance(msg, (HumanMessage, SystemMessage)):
                    filtered_messages.append(msg)
                elif isinstance(msg, ToolMessage):
                    # Convert tool messages to system messages to preserve their content
                    tool_name = getattr(msg, "name", "tool")
                    tool_content = getattr(msg, "content", "")
                    system_msg = SystemMessage(content=f"Tool result from {tool_name}: {tool_content}")
                    filtered_messages.append(system_msg)
                elif getattr(msg, "type", "") in ["human", "system"]:
                    filtered_messages.append(msg)
                elif getattr(msg, "type", "") == "tool":
                    tool_name = getattr(msg, "name", "tool")
                    tool_content = getattr(msg, "content", "")
                    system_msg = SystemMessage(content=f"Tool result from {tool_name}: {tool_content}")
                    filtered_messages.append(system_msg)

            # If no messages after filtering, check for the original human message
            if not filtered_messages:
                for msg in optimized_messages:
                    if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                        filtered_messages.append(msg)
                        break

            # Add the system prompt
            messages_to_send = [self.reasoning_prompt] + filtered_messages

            # Log what we're sending
            print(f"[Agent] Reasoner sending {len(messages_to_send)} messages")
            for i, msg in enumerate(messages_to_send):
                content = getattr(msg, "content", "")
                print(f"[Agent] Message {i}: {type(msg).__name__} - {content[:50]}...")

            # Only invoke once with a timeout
            result = await asyncio.wait_for(
                self.llm_with_tools.ainvoke(
                    messages_to_send,
                    config={"callbacks": [self.streaming_handler]}
                ),
                timeout=60  # Add a timeout to prevent hanging
            )

            # Extract and log the content
            result_text = getattr(result, "content", "").strip().lower()
            print(f"[Agent] Reasoner result preview: {result_text[:100]}...")

            # Determine if we're done based on the magic phrase
            done = "[ready to answer]" in result_text
            print(f"[Agent] Done status: {done}")

            # Flush the thinking buffer to send consolidated thinking content
            await self.streaming_handler._flush_thinking_buffer(done)

            # Update the state with optimized messages and the result
            new_messages = optimized_messages + [result]
            return {**state, "messages": new_messages, "done": done, "step_count": step_count}

        except Exception as e:
            print(f"[Agent] Error in reasoner: {str(e)}")
            # Handle error and still return something valid

            # Flush any buffered thinking content before sending the error
            await self.streaming_handler._flush_thinking_buffer()

            await self.streaming_handler.queue.put({
                "type": "thinking",
                "value": f"Error in reasoning: {str(e)}"
            })

            # Return original messages to avoid corrupting state
            return {**state, "messages": state["messages"], "done": True, "step_count": step_count}  # Force done to exit on error

    async def tools_node(self, state: MessagesState):
        """
        Execute tool calls from the reasoner.

        Args:
            state (dict): Current state with tool calls

        Returns:
            dict: Updated state with tool results
        """
        print("[Agent] Entering tools node")

        # Get the last message which should contain the tool calls
        last_message = state["messages"][-1] if state["messages"] else None

        # Extract tool calls from the message
        tool_calls = []
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_calls = last_message.tool_calls

        # Process each tool call sequentially instead of using ToolNode
        preserved_messages = list(state["messages"])

        # Process tool calls one at a time
        for tool_call in tool_calls:
            tool_name = tool_call.get('name', 'unknown_tool')
            tool_args = tool_call.get('args', {})

            print(f"[Agent] Processing tool call: {tool_name} with args: {tool_args}")

            # Find the matching tool
            matching_tool = None
            for tool in self.tools:
                if hasattr(tool, 'name') and tool.name == tool_name:
                    matching_tool = tool
                    break

            if matching_tool:
                try:
                    # Execute the tool with appropriate arguments
                    if tool_name == "run_command":
                        # Extract command string from arguments
                        command = ""
                        if isinstance(tool_args, dict):
                            if 'command' in tool_args:
                                command = tool_args['command']
                            # Some models may use 'input' instead of 'command'
                            elif 'input' in tool_args:
                                command = tool_args['input']
                            # Handle case where the tool_args might be incorrectly formatted
                            elif len(tool_args) == 1 and 'self' in tool_args:
                                # Try to extract command from the 'self' field if it's a string
                                if isinstance(tool_args['self'], str):
                                    command = tool_args['self']
                                else:
                                    raise ValueError("Invalid command format in tool arguments")
                        elif isinstance(tool_args, str):
                            command = tool_args
                        else:
                            command = str(tool_args)

                        print(f"[Agent] Executing command: {command}")

                        # Use ainvoke with the extracted command
                        result = await matching_tool.ainvoke(command)
                    else:
                        # For other tools, execute with standard arguments
                        # Make sure we handle cases where the tool might be expecting a specific argument name
                        if isinstance(tool_args, dict):
                            # If this is a properly formatted dict with expected argument names, use it directly
                            if not (len(tool_args) == 1 and 'self' in tool_args):
                                result = await matching_tool.ainvoke(tool_args)
                            # If it's a dict with just 'self', extract that value
                            elif isinstance(tool_args['self'], str):
                                result = await matching_tool.ainvoke(tool_args['self'])
                            else:
                                result = await matching_tool.ainvoke({})
                        elif isinstance(tool_args, str):
                            # If it's just a string, pass it directly
                            result = await matching_tool.ainvoke(tool_args)
                        else:
                            # Otherwise, convert to string
                            result = await matching_tool.ainvoke(str(tool_args))

                    # Convert result to a tool message
                    tool_message = ToolMessage(
                        content=result,
                        name=tool_name,
                        tool_call_id=tool_call.get('id', '')
                    )

                    # Also create a system message version for the agent to understand
                    system_msg = SystemMessage(content=f"Tool result from {tool_name}: {result}")

                    # Add both messages to preserved messages
                    preserved_messages.append(system_msg)
                    preserved_messages.append(tool_message)

                    print(f"[Agent] Tool {tool_name} completed with result: {str(result)[:100]}...")

                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    print(f"[Agent] {error_msg}")
                    import traceback
                    traceback.print_exc()

                    # Create error messages
                    error_tool_message = ToolMessage(
                        content=error_msg,
                        name=tool_name,
                        tool_call_id=tool_call.get('id', '')
                    )
                    error_system_msg = SystemMessage(content=f"Tool error from {tool_name}: {error_msg}")

                    # Add error messages
                    preserved_messages.append(error_system_msg)
                    preserved_messages.append(error_tool_message)
            else:
                error_msg = f"Tool {tool_name} not found"
                print(f"[Agent] {error_msg}")

                # Create not found messages
                not_found_tool_message = ToolMessage(
                    content=error_msg,
                    name=tool_name,
                    tool_call_id=tool_call.get('id', '')
                )
                not_found_system_msg = SystemMessage(content=f"Tool error: {error_msg}")

                # Add not found messages
                preserved_messages.append(not_found_system_msg)
                preserved_messages.append(not_found_tool_message)

        # Update the state with preserved messages (preserve step_count)
        return {**state, "messages": preserved_messages}

    async def responder(self, state: MessagesState):
        """
        Generate the final response to the user.

        Args:
            state (dict): Current state with reasoning results

        Returns:
            dict: Updated state with the final response
        """
        print("[Agent] Entering responder node")

        # Flush any remaining thinking content from the reasoner
        await self.streaming_handler._flush_thinking_buffer()

        # Use chat output target for responder - this will show in chat history
        self.streaming_handler.output_target = "chat"

        try:
            # Filter messages for the responder
            # Include the original user query and system messages (including our converted tool results)
            filtered_messages = []
            human_message = None
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    # Keep track of the original human message
                    human_message = msg
                    filtered_messages.append(msg)
                elif isinstance(msg, SystemMessage):
                    filtered_messages.append(msg)
                elif getattr(msg, "type", "") == "human":
                    human_message = msg
                    filtered_messages.append(msg)
                elif getattr(msg, "type", "") == "system":
                    filtered_messages.append(msg)

            # Ensure we have the human message
            if not human_message and len(state["messages"]) > 0:
                # Try to find the first human message in the original state
                for msg in state["messages"]:
                    if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                        human_message = msg
                        if human_message not in filtered_messages:
                            filtered_messages.append(human_message)
                        break

            # Add the response prompt
            messages_to_send = [self.response_prompt] + filtered_messages

            # Log what we're sending to the responder
            print(f"[Agent] Responder sending {len(messages_to_send)} messages")
            for i, msg in enumerate(messages_to_send):
                content = getattr(msg, "content", "")
                print(f"[Agent] Message {i}: {type(msg).__name__} - {content[:50]}...")

            # Generate the final response
            result = await self.llm.ainvoke(
                messages_to_send,
                config={"callbacks": [self.streaming_handler]}
            )

            # No need to manually send events - callback handles it
            new_messages = state["messages"] + [result]
            return {**state, "messages": new_messages}

        except Exception as e:
            print(f"[Agent] Error in responder: {str(e)}")
            # Since this is the last node, output error to chat
            await self.streaming_handler.queue.put({
                "type": "token",
                "value": f"I apologize, but I encountered an error while processing your request. Please try again."
            })
            return state

    def route_from_reasoner(self, state):
        """
        Determine the next step after reasoning.

        Args:
            state (dict): Current state

        Returns:
            str: Next node to execute ("tools" or "responder")
        """
        step_count = state.get('step_count', 0)
        print(f"[Agent] Routing from reasoner. Done: {state.get('done')}, Step: {step_count}")

        # If there was an error or we're explicitly done, go to responder
        if state.get('done', False):
            return "responder"
            
        # Safety check: if we've gone too many steps, force completion
        if step_count >= 35:
            print(f"[Agent] Step limit reached in routing, going to responder")
            return "responder"

        # Check if the last message contains tool calls
        last_message = state["messages"][-1] if state["messages"] else None
        has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls

        if has_tool_calls:
            print(f"[Agent] Found tool calls, going to tools node")
            return "tools"  # Process tool calls
        else:
            # No tool calls but not done yet - go to responder as fallback
            print(f"[Agent] No tool calls, going to responder")
            return "responder"

    async def run_agent_streaming(self, prompt: str):
        """
        Run the agent with streaming output.

        Args:
            prompt (str): The user's prompt/query

        Yields:
            dict: Event objects for streaming (tokens, thinking, tool calls, etc.)
        """
        input_messages = [HumanMessage(content=prompt)]

        # Initialize streaming handler
        self.streaming_handler = StreamingHandler(output_target="debug")

        # Build the agent graph
        self._build_agent_graph()

        async def run_graph():
            print("[Agent] Starting graph execution")
            await self.react_graph.ainvoke(
                {"messages": input_messages, "step_count": 0},
                config={
                    "callbacks": [self.streaming_handler],
                    "recursion_limit": 75  # Increased for complex pentesting scenarios (was 50)
                }
            )
            await self.streaming_handler.end()
            print("[Agent] Graph execution complete")

        graph_task = asyncio.create_task(run_graph())

        try:
            async for event in self.streaming_handler.stream():
                yield event
        finally:
            await graph_task