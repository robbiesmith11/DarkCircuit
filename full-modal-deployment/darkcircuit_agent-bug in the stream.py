import os
import asyncio
import json
import uuid
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.callbacks.base import BaseCallbackHandler
import time
from modal import Dict

from Rag_tool import *

MessagesState = dict


class StreamingHandler(BaseCallbackHandler):
    def __init__(self, output_target="chat"):
        self.queue = asyncio.Queue()
        self.output_target = output_target
        self.buffer = ""
        self.thinking_buffer = ""  # New buffer specifically for thinking tokens
        self.current_phase = None  # Track what phase we're in (reasoning/response)

    async def on_llm_new_token(self, token: str, **kwargs):
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

        # Standardize the format for tool calls
        await self.queue.put({
            "type": "tool_call",
            "name": tool_name,
            "description": tool_description,
            "input": command
        })

    async def on_tool_end(self, output, **kwargs):
        print(f"[StreamingHandler] Tool result: {output}")

        result_content = ""
        # Extract content from different possible formats
        if isinstance(output, ToolMessage):
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

    async def _flush_thinking_buffer(self):
        """Flush accumulated thinking content to the queue"""
        if self.thinking_buffer.strip():
            await self.queue.put({
                "type": "thinking",
                "value": self.thinking_buffer
            })
            self.thinking_buffer = ""  # Clear the buffer after sending

    async def stream(self):
        while True:
            item = await self.queue.get()
            if item == "__END__":
                break
            yield item

    async def end(self):
        # Make sure to flush any remaining thinking content
        await self._flush_thinking_buffer()

        if self.buffer.strip():
            await self.queue.put({"type": "token", "value": self.buffer})
        await self.queue.put("__END__")


class Darkcircuit_Agent:
    def __init__(self):

        api_key = os.environ["OPENAI_API_KEY"]

        self.llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, api_key=api_key)
        self.streaming_handler = None
        self.terminal_output_queue = asyncio.Queue()
        self.terminal_command_id = 0
        self.current_command_output = ""

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
            """Execute a command on the remote SSH server through the UI terminal."""
            try:
                # Generate a unique ID for this command execution
                self.terminal_command_id += 1
                command_id = self.terminal_command_id

                # Tell the frontend to execute the command
                await self.streaming_handler.queue.put({
                    "type": "ui_terminal_command",
                    "command": command,
                    "command_id": command_id
                })

                # Wait for output using the file-based approach
                return await self._wait_for_terminal_output(command_id)

            except Exception as e:
                return f"Error executing command: {str(e)}"

        self.run_command = run_command
        self.rag_retrieve = rag_retrieve
        self.tools = [self.search, self.run_command, self.rag_retrieve]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.reasoning_prompt = SystemMessage(content="""You are a multi-step problem solver. Always follow this pattern:

        1. Analyze the user request.
        2. Decide if a tool is needed (search or command).
        3. Use the tool and analyze the result.
        4. ONLY when you have everything you need and are fully prepared to give the final answer, conclude with the exact phrase: 'Ready to answer.'

        IMPORTANT: 
        - Do NOT use the phrase 'Ready to answer' anywhere in your thinking process except as the final signal.
        - Do NOT output the final answer here - only think through the steps.
        - Do NOT repeat the instructions or the 'Ready to answer' phrase when outlining your approach.
        - If you need to use a tool, clearly indicate which tool you want to use and what input you're providing.
        - Avoid repeating tool actions indefinitely. If a tool result is unclear or incomplete after 3 tries, stop and respond.
        - If a command might run forever (like 'ping'), make sure it has a limit (e.g., 'ping -c 4').
        - For network scanning commands like nmap that can take a long time, consider adding the --min-rate parameter to speed up scanning.

        Hack The Box Challenges:
        - If the user asks to analyze, enumerate, or exploit a Hack The Box machine (e.g., "Start on Dancing at <target_ip>"):
            - Use your own knowledge and the RAG tool to gather relevant context about the machine.
            - Determine which recon or exploit commands would help investigate the machine based on its name, known ports, or CVEs.
            - Use the 'run_command' tool to execute those commands automatically over SSH.
            - You may run multiple useful commands in sequence without asking for confirmation.
            - Always analyze each command's output before deciding what to do next.
            - Keep safety in mind and avoid dangerous commands like `rm`, `shutdown`, `:(){ :|: & };:` or infinite loops.

        Begin your analysis now.
        """)
        self.response_prompt = SystemMessage(
            content="Now answer the user's question clearly and concisely based on previous analysis and tool results.")


        self.checkpointer = MemorySaver()  # Or RedisSaver(), SQLiteSaver() etc.
        builder = StateGraph(MessagesState)
        builder.add_node("reasoner", self.reasoner)
        builder.add_node("tools", self.tools_node)  # Custom tools node to handle tool results
        builder.add_node("responder", self.responder)
        builder.add_edge(START, "reasoner")
        builder.add_conditional_edges("reasoner", self.route_from_reasoner)
        builder.add_edge("tools", "reasoner")
        builder.add_edge("responder", END)

        self.react_graph = builder.compile(checkpointer=self.checkpointer)

    # Define the method properly at the class level
    async def _wait_for_terminal_output(self, command_id):
        """Wait for terminal output using Modal's shared Dict with timestamp validation."""
        session_start_time = time.time()
        command_id_str = str(command_id)

        # Get a reference to the shared dict
        command_results = Dict.from_name("terminal-command-results")

        # Track partial outputs
        last_output = ""
        last_output_time = time.time()
        seen_partial = False

        print(f"Waiting for output of command {command_id}")

        while True:
            # Check timeouts
            if time.time() - session_start_time > 300:  # 5 minute absolute timeout
                return f"Command timed out after 5 minutes. Last output:\n\n{last_output}"

            if seen_partial and time.time() - last_output_time > 60:  # 1 minute without updates
                return f"Command appears to have stalled. Last output:\n\n{last_output}"

            # Check if result exists in the shared dict
            if command_id_str in command_results:
                result_data = command_results[command_id_str]

                # Check if this is a proper dict with timestamp (new format)
                if isinstance(result_data, dict) and "timestamp" in result_data:
                    # Only use results from this session
                    if result_data["timestamp"] >= session_start_time:
                        result = result_data["output"]
                        print(f"Found result for command {command_id}, length: {len(result)}")

                        # Process result...
                        partial_marker = "[Command still running... This is a partial output.]"
                        if partial_marker in result:
                            # Partial output handling...
                            seen_partial = True
                            if len(result) > len(last_output):
                                last_output = result
                                last_output_time = time.time()
                            await asyncio.sleep(1.0)
                            continue
                        else:
                            # This is the final result
                            print(f"Received final output for command {command_id}")
                            # Clean up
                            try:
                                del command_results[command_id_str]
                            except:
                                pass
                            await asyncio.sleep(0.5)
                            return result
                    else:
                        # This is an old result from a previous session, ignore it
                        print(f"Ignoring stale result for command {command_id} from previous session")
                        try:
                            del command_results[command_id_str]
                        except:
                            pass
                else:
                    # Old format (just string) or invalid data - handle for backward compatibility
                    if isinstance(result_data, str):
                        # Treat as final output and clean up
                        print(f"Found old-format result for command {command_id}")
                        result = result_data
                        try:
                            del command_results[command_id_str]
                        except:
                            pass
                        await asyncio.sleep(0.5)
                        return result
                    else:
                        # Invalid data, clean up
                        print(f"Found invalid data for command {command_id}, cleaning up")
                        try:
                            del command_results[command_id_str]
                        except:
                            pass

            # Wait before checking again
            await asyncio.sleep(0.5)

    # Define the receive method properly at the class level
    async def receive_terminal_output(self, output_data):
        """Receive terminal output from the frontend.

        This method is called by the backend when it receives terminal output
        from the frontend through the /api/terminal/output endpoint.
        """
        await self.terminal_output_queue.put(output_data)
        #print(output_data)
        #print(self.terminal_output_queue.get())

        # Also store the latest output for the current command
        self.current_command_output = output_data.get("output", "")

    async def reasoner(self, state: MessagesState):
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
            done = "ready to answer" in result_text
            print(f"[Agent] Done status: {done}")

            # Flush the thinking buffer to send consolidated thinking content
            await self.streaming_handler._flush_thinking_buffer()

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
        """Custom tools node to handle tool calls and preserve results as system messages"""
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
                    # Execute the tool with its arguments
                    if tool_name == "run_command":
                        # Special handling for run_command to ensure sequential execution
                        # Extract command from arguments, handling different formats
                        if isinstance(tool_args, dict) and 'command' in tool_args:
                            command = tool_args['command']
                        elif isinstance(tool_args, str):
                            command = tool_args
                        else:
                            command = str(tool_args)

                        print(f"[Agent] Executing command: {command}")

                        # Add optimization hints for common long-running commands
                        if 'nmap' in command and '--min-rate' not in command:
                            # Check if this is a comprehensive scan that will take a long time
                            if any(flag in command for flag in ['-sS', '-sV', '-A', '-p-']):
                                # Add optimized version with hint
                                command_original = command
                                # Add min-rate parameter if not already present
                                command = command + ' --min-rate=1000' if '--min-rate' not in command else command
                                print(f"[Agent] Optimized slow nmap command: {command_original} -> {command}")

                                # Add message about optimization
                                system_msg = SystemMessage(
                                    content=f"Optimizing potentially slow nmap command with --min-rate=1000 to speed up execution.")
                                preserved_messages.append(system_msg)

                        # Properly invoke the async method
                        # The key change is using ainvoke instead of calling run_command directly
                        result = await matching_tool.ainvoke(command)
                    else:
                        # For other tools, execute normally
                        if isinstance(tool_args, dict):
                            # For dict args, use the appropriate invocation
                            result = await matching_tool.ainvoke(tool_args)
                        else:
                            # For simple string args
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
        print(f"[Agent] Routing from reasoner. Done: {state.get('done')}")

        # If there was an error or we're explicitly done, go to responder
        if state.get('done', False):
            return "responder"

        # Check if the last message contains tool calls
        last_message = state["messages"][-1] if state["messages"] else None
        has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls

        if has_tool_calls:
            return "tools"  # Process tool calls
        else:
            # No tool calls but not done yet - should we try more reasoning?
            # For now, just go to responder as fallback to avoid getting stuck
            return "responder"

    from langchain_core.messages import HumanMessage

    async def run_agent_streaming(self, prompt: str):
        import uuid
        session_id = "debug-session"  # You can use str(uuid.uuid4()) for per-run memory

        self.streaming_handler = StreamingHandler(output_target="debug")

        async def run_graph():
            print("[Agent] Starting graph execution")

            try:
                print("[Agent] Initial checkpoints:", list(self.checkpointer.list({})))

                # Default state input if memory fails
                state_input = {"messages": [HumanMessage(content=prompt)]}

                # Try to restore state from memory
                restored_state = self.checkpointer.get(session_id)

                def is_valid_checkpoint(state):
                    try:
                        return (
                                isinstance(state, dict)
                                and isinstance(state.get("channel_values", {}), dict)
                                and isinstance(state["channel_values"].get("__root__", {}), dict)
                                and "messages" in state["channel_values"]["__root__"]
                                and isinstance(state["channel_values"]["__root__"]["messages"], list)
                        )
                    except Exception:
                        return False

                if is_valid_checkpoint(restored_state):
                    try:
                        restored_messages = restored_state["channel_values"]["__root__"]["messages"]
                        if restored_messages:
                            print(f"[Agent] ✅ Restored {len(restored_messages)} messages for session {session_id}")
                            state_input = {}  # Let LangGraph handle the restoration internally
                        else:
                            raise ValueError("Checkpoint contains no messages.")
                    except Exception as e:
                        print(f"[Agent] ⚠️ Invalid checkpoint structure: {e}")
                        self.checkpointer.delete(session_id)
                else:
                    print("[Agent] 🧹 Wiping invalid or missing checkpoint")
                    self.checkpointer.delete(session_id)

                await self.react_graph.ainvoke(
                    state_input,
                    config={
                        "callbacks": [self.streaming_handler],
                        "configurable": {
                            "thread_id": session_id
                        }
                    }
                )

                print("[Agent] ✅ Graph execution completed. Current checkpoints:", list(self.checkpointer.list({})))

            except Exception as e:
                print(f"[Agent] ❌ Error during graph execution: {e}")
            finally:
                await self.streaming_handler.end()
                print("[Agent] 🛑 Graph execution finished")

        # Launch the graph in a background task and stream output
        graph_task = asyncio.create_task(run_graph())

        try:
            async for event in self.streaming_handler.stream():
                yield event
        finally:
            await graph_task

