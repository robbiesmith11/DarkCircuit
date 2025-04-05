import os
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage


os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI(model="gpt-3.5-turbo")
search = DuckDuckGoSearchRun()
tools = [search]
llm_with_tools = llm.bind_tools(tools)


system_message = SystemMessage(
    content="You are an AI assistant tasked with helping answer users' questions. You can use tools like search to find information on the web."
)

# Define the reasoning function for the agent
def reasoner(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([system_message] + state["messages"])]}

# Build the reactive graph using LangGraph
builder = StateGraph(MessagesState)
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "reasoner")
builder.add_conditional_edges("reasoner", tools_condition)
builder.add_edge("tools", "reasoner")
react_graph = builder.compile()

def run_agent(prompt: str) -> dict:
    """
    Process a prompt using the agent and return a dictionary with a 'messages'
    key containing a list of message dictionaries with 'role' and 'content'.
    """
    # Create input as a HumanMessage
    input_messages = [HumanMessage(content=prompt)]
    output_state = react_graph.invoke({"messages": input_messages})
    # Convert each message object to a dict; hardcode role as "assistant"
    messages = [{"role": "assistant", "content": msg.content} for msg in output_state["messages"]]

    return {"messages": messages}