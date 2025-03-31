
import os
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from tools import multiply


def reasoner(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([system_message] + state["messages"])]}


#os.environ["OPENAI_API_KEY"] = "your key
llm = ChatOpenAI(model="gpt-3.5-turbo")

search = DuckDuckGoSearchRun()

#add tools to the model to use
tools = [multiply, search]
llm_with_tools = llm.bind_tools(tools)

system_message = SystemMessage(content="You are an AI assisstant tasked with helping answer users questions you are able to use tools such as multiple and search to find anwsers on the web")

#graph design
builder = StateGraph(MessagesState)
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "reasoner")
builder.add_conditional_edges("reasoner", tools_condition)
builder.add_edge("tools", "reasoner")
react_graph = builder.compile()


def run_agent(prompt: str) -> str:

    input_messages = [HumanMessage(content=prompt)]
    output_state = react_graph.invoke({"messages": input_messages})

    return output_state


#user enter prompt
user_prompt = input("Enter your question")
response = run_agent(user_prompt)

#display output
for msg in response["messages"]:
    msg.pretty_print()

