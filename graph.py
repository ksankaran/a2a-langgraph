from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langgraph.prebuilt import ToolNode, tools_condition
from tools import get_forecast

examples = [
    HumanMessage(
        "what is the weather forecast for cumming ga?", name="example_user"
    ),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {"name": "get_forecast", "args": {"lat": "34.208464", "long": "-84.137575"}, "id": "1"}
        ],
    ),
    ToolMessage("It is sunny in Cumming, GA", tool_call_id="1"),
    AIMessage(
        "I have successfully obtained the forecast for Cumming, GA. It is sunny there.",
        name="example_assistant",
    ),
]

system = """I will help user with weather information using the tool provided.
You will be provided with a tool to get the weather forecast based on latitude and longitude.
You can use the tool by calling it with the latitude and longitude values.

You will receive a query from the user, and if it is related to weather forecast, use the tool to provide the weather information.
If not, respond with a message indicating that you can only provide weather information."""

few_shot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        *examples,
        ("human", "{query}"),
    ]
)

tools = [
    get_forecast,
]

class State(TypedDict):
    """
    Represents the state of the agent.
    """
    messages: Annotated[list[AnyMessage], add_messages]

def chatbot(state: State) -> str:
    """
    Contacts LLM to get a response based on the current state.
    Args:
        state (State): The current state of the agent.
    Returns:
        state (State): The updated state with the response.
    """
    params = {
        "azure_endpoint": "https://mygateway.example.com",
        "azure_deployment": "gpt-4o-mini_2024-07-18",
        "api_version": "version-override",
        "api_key": "api_key",
        "timeout": 60,
    }
    model = AzureChatOpenAI(**params)
    model_with_tools = model.bind_tools(tools)
    chain = {"query": RunnablePassthrough()} | few_shot_prompt | model_with_tools
    
    messages = state["messages"]
    response = chain.invoke(messages)
    return { "messages": [response] }

builder = StateGraph(State)

builder.add_edge(START, 'chatbot')
builder.add_node("tools", ToolNode(tools))
builder.add_node('chatbot', chatbot)
builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
builder.add_edge("tools", "chatbot")
builder.set_entry_point("chatbot")
builder.add_edge('chatbot', END)

graph = builder.compile()