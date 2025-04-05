import os
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
import paramiko


class Darkcircuit_Agent:
    def __init__(self, client=None, **kwargs):
        # Set OpenAI API key
        os.environ[
            "OPENAI_API_KEY"] = "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"
            "OPENAI_API_KEY"] = "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"

        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")

        # Initialize SSH client
        self.ssh_client = client

        # Define tools
        self.search = DuckDuckGoSearchRun()

        # Define the run_command tool
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

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Create system message
        self.system_message = SystemMessage(
            content="You are an AI assistant tasked with helping answer users' questions. You can use tools like search to find information on the web, and run commands on a remote server if connected."
        )

        # Define the reasoning function
        def reasoner(state: MessagesState):
            return {"messages": [self.llm_with_tools.invoke([self.system_message] + state["messages"])]}

        # Build the reactive graph
        builder = StateGraph(MessagesState)
        builder.add_node("reasoner", reasoner)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "reasoner")
        builder.add_conditional_edges("reasoner", tools_condition)
        builder.add_edge("tools", "reasoner")
        self.react_graph = builder.compile()

    def run_agent(self, prompt: str) -> dict:
        """
        Process a prompt using the agent and return a dictionary with a 'messages'
        key containing a list of message dictionaries with 'role' and 'content'.
        """
        # Create input as a HumanMessage
        input_messages = [HumanMessage(content=prompt)]
        output_state = self.react_graph.invoke({"messages": input_messages})

        # Convert each message object to a dict
        messages = [{"role": "assistant", "content": msg.content} for msg in output_state["messages"]]

        return {"messages": messages}

    def __del__(self):
        """Close SSH connection when the object is destroyed"""
        if self.ssh_client:
            self.ssh_client.close()


# Example usage
if __name__ == "__main__":
    # Create agent with or without SSH connection
    # Without SSH:
    agent = Darkcircuit_Agent()

    # With SSH:
    # agent = Darkcircuit_Agent(ssh_state["client"])

    result = agent.run_agent("What's the weather in New York today?")
    print(result["messages"])