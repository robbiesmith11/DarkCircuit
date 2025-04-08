import os
import asyncio
import uuid
import paramiko
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.callbacks.base import BaseCallbackHandler

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

        tool_name = getattr(tool, "name", str(tool))
        print(f"[StreamingHandler] Starting tool: {tool_name} with input: {input_str}")

        # Standardize the format for tool calls
        await self.queue.put({
            "type": "tool_call",
            "name": tool_name,
            "input": input_str
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
    def __init__(self, client=None):
        api_key = os.environ["OPENAI_API_KEY"]

        self.llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, api_key=api_key)
        self.ssh_client = client
        self.streaming_handler = None

        self.search = DuckDuckGoSearchRun()

        @tool
        def run_command(command: str) -> str:
            """Execute a command on the remote SSH server"""
            if self.ssh_client is None:
                return "Error: SSH client not connected"
            try:
                stdin, stdout, stderr = self.ssh_client.exec_command(command, timeout=30)
                output = stdout.read().decode('utf-8', errors='replace')
                error = stderr.read().decode('utf-8', errors='replace')
                return f"Output:\n{output}\n\nErrors:\n{error}" if error else output
            except Exception as e:
                return f"Error executing command: {str(e)}"

        self.run_command = run_command
        self.tools = [self.search, self.run_command]
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

        Begin your analysis now.
        """)
        self.response_prompt = SystemMessage(
            content="Now answer the user's question clearly and concisely based on previous analysis and tool results.")

        builder = StateGraph(MessagesState)
        builder.add_node("reasoner", self.reasoner)
        builder.add_node("tools", self.tools_node)  # Custom tools node to handle tool results
        builder.add_node("responder", self.responder)
        builder.add_edge(START, "reasoner")
        builder.add_conditional_edges("reasoner", self.route_from_reasoner)
        builder.add_edge("tools", "reasoner")
        builder.add_edge("responder", END)

        self.react_graph = builder.compile()

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

        # Get the standard ToolNode to process the tool calls
        tool_node = ToolNode(self.tools)
        result_state = await tool_node.ainvoke(state)

        # Extract the tool results and convert them to system messages
        preserved_messages = []
        for msg in result_state["messages"]:
            if isinstance(msg, ToolMessage):
                # Convert tool message to system message
                tool_name = getattr(msg, "name", "tool")
                tool_content = getattr(msg, "content", "")
                system_msg = SystemMessage(content=f"Tool result from {tool_name}: {tool_content}")
                preserved_messages.append(system_msg)
                # Also keep the original tool message for the LangGraph machinery
                preserved_messages.append(msg)
            else:
                preserved_messages.append(msg)

        # Update the state with preserved messages
        return {**result_state, "messages": preserved_messages}

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

    async def run_agent_streaming(self, prompt: str):
        input_messages = [HumanMessage(content=prompt)]

        # Initialize streaming handler
        self.streaming_handler = StreamingHandler(output_target="debug")

        async def run_graph():
            print("[Agent] Starting graph execution")
            await self.react_graph.ainvoke(
                {"messages": input_messages},
                config={"callbacks": [self.streaming_handler]}
            )
            await self.streaming_handler.end()
            print("[Agent] Graph execution complete")

        graph_task = asyncio.create_task(run_graph())

        try:
            async for event in self.streaming_handler.stream():
                yield event
        finally:
            await graph_task

    def __del__(self):
        if self.ssh_client:
            self.ssh_client.close()