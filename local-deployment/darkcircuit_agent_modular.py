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
                 model_name="gpt-3.5-turbo",  # Default to cheaper model
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
        
        # Keep track of conversation history
        self.chat_history = []
        self.max_history_length = 10  # Keep at 10 messages to ensure sufficient context

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
        def rag_retrieve_with_context(query: str) -> str:
            """
            [CONTEXT-AWARE RAG] Search for relevant documents using conversation history to improve results.
            Use this tool for follow-up questions or when the query is related to previous conversation.
            """
            from Rag_tool import rag_retrieve_with_history
            
            # Extract recent conversation history from the agent's state
            chat_history = []
            if hasattr(self, 'chat_history') and self.chat_history:
                chat_history = self.chat_history
            
            # Use the history-aware retriever
            result = rag_retrieve_with_history(
                query=query, 
                chat_history=chat_history,
                model_name="gpt-4o-mini"  # Using cost-effective model for RAG results
            )
            
            # Add a clear indicator at the beginning of the response
            return "🧠 [CONTEXT-AWARE SEARCH RESULTS] Using conversation history to enhance retrieval:\n\n" + result

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
        self.rag_retrieve_with_context = rag_retrieve_with_context
        self.tools = [self.search, self.run_command, self.rag_retrieve, self.rag_retrieve_with_context]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Load default prompts from file or use hardcoded defaults
        DEFAULT_REASONING_PROMPT, DEFAULT_RESPONSE_PROMPT = load_prompts()

        # Create system messages
        self.reasoning_prompt = SystemMessage(content=reasoning_prompt or DEFAULT_REASONING_PROMPT)
        self.response_prompt = SystemMessage(content=response_prompt or DEFAULT_RESPONSE_PROMPT)

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

        # Use debug output target for reasoner
        self.streaming_handler.output_target = "debug"

        try:
            # When preparing messages for the reasoner, include the original user query
            # and system messages, but convert tool messages to system messages
            filtered_messages = []
            for msg in state["messages"]:
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
                for msg in state["messages"]:
                    if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                        filtered_messages.append(msg)
                        break

            # Enhance with learning recommendations if available
            reasoning_prompt_content = self.reasoning_prompt.content
            
            # Add learning recommendations if available
            if hasattr(self, 'get_learning_recommendations'):
                try:
                    recommendations = self.get_learning_recommendations()
                    
                    # Create an addition to the prompt with learned techniques
                    learned_techniques = []
                    
                    # Add flag techniques
                    if recommendations.get("flag_techniques"):
                        techniques = recommendations["flag_techniques"]
                        learned_techniques.append("PREVIOUSLY SUCCESSFUL FLAG-FINDING TECHNIQUES:")
                        for i, technique in enumerate(techniques):
                            description = technique.get("description", "")
                            key_commands = technique.get("key_commands", [])
                            learned_techniques.append(f"{i+1}. {description}")
                            if key_commands:
                                learned_techniques.append("   Key commands:")
                                for cmd in key_commands:
                                    learned_techniques.append(f"   - {cmd}")
                                    
                    # Add recommended next steps
                    if recommendations.get("next_steps"):
                        next_steps = recommendations["next_steps"]
                        if next_steps:
                            learned_techniques.append("\nRECOMMENDED NEXT COMMANDS:")
                            for i, cmd in enumerate(next_steps):
                                learned_techniques.append(f"{i+1}. {cmd}")
                    
                    # Add similar commands that worked before
                    if recommendations.get("similar_commands"):
                        similar_commands = recommendations["similar_commands"]
                        if similar_commands:
                            learned_techniques.append("\nSIMILAR COMMANDS THAT WORKED BEFORE:")
                            for i, cmd in enumerate(similar_commands):
                                learned_techniques.append(f"{i+1}. {cmd}")
                    
                    # Only add to prompt if we have recommendations
                    if learned_techniques:
                        learning_addition = "\n\n# LEARNED FROM PREVIOUS SUCCESSES\n" + "\n".join(learned_techniques)
                        reasoning_prompt_content += learning_addition
                        print(f"[Agent] Enhanced prompt with {len(learned_techniques)} learned techniques")
                except Exception as e:
                    print(f"[Agent] Error enhancing prompt with learning: {e}")
            
            # Create an enhanced system message with learning
            enhanced_system_prompt = SystemMessage(content=reasoning_prompt_content)
            
            # Add the system prompt
            messages_to_send = [enhanced_system_prompt] + filtered_messages

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

            # Count how many commands were run by looking at tool calls in the history
            command_count = 0
            for msg in state["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.get('name') == 'run_command':
                            command_count += 1
            
            # Custom logic for determining done status
            # Only stop if we explicitly see the ready signal AND have run enough commands
            # or if we've run a very high number of commands (prevent infinite loops)
            done = "[ready to answer]" in result_text
            
            # If in HTB/security context, enforce minimum command count
            is_security_context = any(term in result_text.lower() for term in 
                                    ["hack the box", "htb", "exploit", "nmap", "vulnerability", 
                                     "security", "ssh", "brute force", "password"]) or \
                               any(term.lower() in self.last_query.lower() if hasattr(self, 'last_query') and self.last_query else False 
                                   for term in ["hack the box", "htb", "exploit", "flag", "challenge", "lab", "target"])
            
            # Check for flag patterns in the entire conversation history
            has_flag = False
            for msg in state["messages"]:
                content = getattr(msg, "content", "").lower()
                if any(pattern in content for pattern in ["[flag", "flag:", "htb{", "root:", "user:", "flag{"]) and \
                   any(indicator in content for indicator in ["found", "discovered", "got", "here", "is"]):  
                    has_flag = True
                    break
            
            # Higher command count minimum for security context
            REQUIRED_COMMANDS = 20 if is_security_context else 10
            
            # Enforce strict minimum command count for security contexts
            if is_security_context and command_count < REQUIRED_COMMANDS and not has_flag:
                # Force continued execution if not enough commands have been tried
                force_execution = True
                done = False
                print(f"[Agent] Only {command_count}/{REQUIRED_COMMANDS} required commands executed, continuing execution")
            
            # In security contexts, NEVER stop unless flag found or very high command count
            if is_security_context and not has_flag and command_count < 50:  # Absolute upper limit is 50 commands
                done = False  # Keep going until we find the flag
                print(f"[Agent] Security context detected and flag not found. Continuing regardless of ready signal.")
            
            # If we don't see tool calls in the result, but we're not done, check if the model is just thinking
            if not done and not hasattr(result, "tool_calls") and "[ready to answer]" not in result_text and is_security_context:
                # Check the text for indicators it's asking for permission or planning next steps
                if any(phrase in result_text for phrase in ["should i", "shall i", "would you", "do you want", "next step", "proceed"]):
                    # Enforce tool usage rather than thinking by overriding the result
                    # Create a default command to force execution
                    print(f"[Agent] Detected planning without action. Forcing command execution.")
                    
                    # Determine a good default command based on context
                    default_cmd = "whoami && pwd && ls -la"
                    if "web" in result_text or "http" in result_text:
                        default_cmd = "curl -v http://localhost/ || curl -v http://127.0.0.1/"
                    elif "port" in result_text or "scan" in result_text:
                        default_cmd = "nmap -p- --min-rate 5000 -T4 127.0.0.1 || nmap -p- --min-rate 5000 -T4 localhost"
                    
                    # Create a tool call to force execution
                    from langchain_core.messages import AIMessage
                    result = AIMessage(
                        content="I'll run a command to gather more information.",
                        tool_calls=[{"name": "run_command", "args": {"command": default_cmd}}]
                    )
                    done = False
            
            print(f"[Agent] Done status: {done}")

            # Flush the thinking buffer to send consolidated thinking content
            await self.streaming_handler._flush_thinking_buffer(done)

            # Update the state with filtered messages and the result
            new_messages = filtered_messages + [result]
            return {**state, "messages": new_messages, "done": done}

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
            return {**state, "messages": state["messages"], "done": True}  # Force done to exit on error

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

        # Update the state with preserved messages
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
        print(f"[Agent] Routing from reasoner. Done: {state.get('done')}")

        # Get last message content for context
        last_message = state["messages"][-1] if state["messages"] else None
        message_content = getattr(last_message, "content", "").lower() if last_message else ""
        has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls

        # Check for questions about whether to proceed
        asks_to_proceed = any(phrase in message_content for phrase in 
                           ["shall we proceed", "should we proceed", "would you like to proceed", 
                            "do you want me to", "should i proceed", "shall i proceed", "let me know", 
                            "what would you like", "next step", "would you like", "how should i"])

        # Is this a security context?
        is_security_context = False
        for msg in state["messages"]:
            content = getattr(msg, "content", "").lower()
            if any(term in content for term in ["hack the box", "htb", "exploit", "vulnerability", 
                                             "security", "ssh", "brute force", "password", "flag", "ctf"]):
                is_security_context = True
                break

        # If asking for permission to continue OR in security context and not making progress, force execution
        force_execution = asks_to_proceed or (
            is_security_context and 
            not has_tool_calls and 
            not state.get('done', False) and
            "[ready to answer]" not in message_content
        )
            
        if force_execution:
            print(f"[Agent] Detected need for forced execution - automatically running command")
            # Force execution to continue by routing to tools node
            if not has_tool_calls:
                # Create a dummy tool call message to force execution
                from langchain_core.messages import AIMessage
                
                # Determine a good default command based on context
                default_cmd = "whoami && pwd && ls -la"
                if "web" in message_content or "http" in message_content:
                    default_cmd = "curl -v http://localhost/ || curl -v http://127.0.0.1/"
                elif "port" in message_content or "scan" in message_content:
                    default_cmd = "nmap -p- --min-rate 5000 -T4 127.0.0.1 || nmap -p- --min-rate 5000 -T4 localhost"
                
                tool_msg = AIMessage(
                    content="Let's proceed with further enumeration and exploitation",
                    tool_calls=[{"name": "run_command", "args": {"command": default_cmd}}]
                )
                state["messages"].append(tool_msg)
                state["done"] = False
                return "tools"

        # If there was an error or we're explicitly done, go to responder
        if state.get('done', False):
            return "responder"

        # Check if the last message contains tool calls
        if has_tool_calls:
            return "tools"  # Process tool calls
        else:
            # No tool calls but not done yet - go to responder as fallback
            return "responder"

    async def run_agent_streaming(self, prompt: str):
        """
        Run the agent with streaming output.

        Args:
            prompt (str): The user's prompt/query

        Yields:
            dict: Event objects for streaming (tokens, thinking, tool calls, etc.)
        """
        # Store the last user query for context detection
        self.last_query = prompt
        
        # Add user message to chat history
        user_message = HumanMessage(content=prompt)
        self.chat_history.append(user_message)
        
        # Maintain a fixed size history window
        if len(self.chat_history) > self.max_history_length:
            self.chat_history = self.chat_history[-self.max_history_length:]
        
        input_messages = [user_message]

        # Initialize streaming handler
        self.streaming_handler = StreamingHandler(output_target="debug")

        # Build the agent graph
        self._build_agent_graph()

        async def run_graph():
            print("[Agent] Starting graph execution")
            result = await self.react_graph.ainvoke(
                {"messages": input_messages},
                config={"callbacks": [self.streaming_handler]}
            )
            await self.streaming_handler.end()
            print("[Agent] Graph execution complete")
            
            # Extract the AI's final answer to add to chat history
            if result and "messages" in result:
                final_messages = result["messages"]
                if final_messages and len(final_messages) > 0:
                    last_message = final_messages[-1]
                    if hasattr(last_message, "content") and not isinstance(last_message, HumanMessage):
                        # Add AI message to chat history
                        ai_message = AIMessage(content=last_message.content)
                        self.chat_history.append(ai_message)
                        
                        # Maintain a fixed size history window
                        if len(self.chat_history) > self.max_history_length:
                            self.chat_history = self.chat_history[-self.max_history_length:]

        graph_task = asyncio.create_task(run_graph())

        try:
            async for event in self.streaming_handler.stream():
                yield event
        finally:
            await graph_task