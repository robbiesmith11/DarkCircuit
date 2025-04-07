import os
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.callbacks.base import BaseCallbackHandler
import asyncio
import paramiko


class StreamingHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.queue.put({"type": "token", "value": token})

    async def on_tool_start(self, tool, input_str, **kwargs):
        await self.queue.put({
            "type": "tool_call",
            "name": tool.name if hasattr(tool, "name") else str(tool),
            "input": input_str
        })

    async def on_tool_end(self, output, **kwargs):
        if isinstance(output, ToolMessage):
            output = {
                "tool_call_id": output.tool_call_id,
                "content": output.content
            }
        await self.queue.put({
            "type": "tool_result",
            "output": output
        })

    async def stream(self):
        while True:
            item = await self.queue.get()
            if item == "__END__":
                break
            yield item

    async def end(self):
        await self.queue.put("__END__")



class Darkcircuit_Agent:
    def __init__(self, client=None, **kwargs):
        # Set OpenAI API key
        os.environ[
            "OPENAI_API_KEY"] = <replace with your OpenAI API key>

        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

        # Initialize SSH client
        self.ssh_client = client

        # Define tools
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

                if error:
                    return f"Output:\n{output}\n\nErrors:\n{error}"
                return output
            except Exception as e:
                return f"Error executing command: {str(e)}"

        self.run_command = run_command
        self.tools = [self.search, self.run_command]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.system_message = SystemMessage(
            content="You are an AI assistant tasked with helping answer users' questions. You can use tools like search to find information on the web, and run commands on a remote server if connected."
        )

        # ✅ ASYNC reasoner function with proper streaming
        async def reasoner(state: MessagesState):
            response = await self.llm_with_tools.ainvoke(
                [self.system_message] + state["messages"],
                config={"callbacks": [self.streaming_handler]} if self.streaming_handler else None
            )
            return {"messages": [response]}

        # Build the LangGraph
        builder = StateGraph(MessagesState)
        builder.add_node("reasoner", reasoner)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "reasoner")
        builder.add_conditional_edges("reasoner", tools_condition)
        builder.add_edge("tools", "reasoner")
        self.react_graph = builder.compile()

    def run_agent(self, prompt: str) -> dict:
        input_messages = [HumanMessage(content=prompt)]
        output_state = self.react_graph.invoke({"messages": input_messages})
        messages = [{"role": "assistant", "content": msg.content} for msg in output_state["messages"]]
        return {"messages": messages}

    async def run_agent_streaming(self, prompt: str):
        input_messages = [HumanMessage(content=prompt)]
        handler = StreamingHandler()
        self.streaming_handler = handler

        final_output = {}

        async def run_graph():
            nonlocal final_output
            final_output = await self.react_graph.ainvoke(
                {"messages": input_messages},
                config={"callbacks": [handler]}
            )
            await handler.end()

        # Start graph and stream simultaneously
        graph_task = asyncio.create_task(run_graph())
        async for event in handler.stream():
            yield event
        await graph_task  # wait for graph to complete

        # If LLM responded after tools, yield that too
        final_messages = final_output.get("messages", [])
        final_msg = next((m for m in reversed(final_messages) if m.type == "ai"), None)
        if final_msg:
            yield {"type": "token", "value": final_msg.content}

    def __del__(self):
        if self.ssh_client:
            self.ssh_client.close()


# Manual test (non-streaming)
if __name__ == "__main__":
    agent = Darkcircuit_Agent()
    result = agent.run_agent("What's the weather in New York today?")
    print(result["messages"])
