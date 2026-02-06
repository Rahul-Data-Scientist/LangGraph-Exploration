# There is a problem with sir's chatbot_with_hitl code. Sir has implemented HITL inside a tool which is not desired. A tool is supposed to perform a specific task only. It must not handle workflow situations like HITL.
# So, in this code, we will make another node by the name 'human_review_node' whcih will handle the HITL

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode, tools_condition


from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from typing import TypedDict, Annotated

from dotenv import load_dotenv

import requests

import os

load_dotenv()

stock_api_key = os.environ["STOCK_API_KEY"]

search_tool = DuckDuckGoSearchRun(region = "us-en")

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stock_api_key}"
    r = requests.get(url)
    return r.json()

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """Simulate purchasing a given quantity of a stock symbol."""
    return {
        "status": "success",
        "message": f"Successfully purchased {quantity} of {symbol}."
    }

tools = [search_tool, get_stock_price, purchase_stock]

llm = ChatOpenAI(model = "gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

tool_node = ToolNode(tools)

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def human_review_node(state: ChatState):
    """
    Acts as a gateway. If 'purchase_stock' is called, it interrupts for human approval.
    """
    
    last_message = state['messages'][-1]

    purchase_call = [tc for tc in last_message.tool_calls if tc['name'] == 'purchase_stock']

    if purchase_call:
        tool_call = purchase_call[0]
        symbol = tool_call['args']['symbol']
        quantity = tool_call['args']['quantity']
        tool_call_id = tool_call['id']

        decision = interrupt(f"CONFIRMATION REQUIRED: Purchase {quantity} stocks of {symbol}?")

        if isinstance(decision, str) and decision.strip().lower() != "yes":
            return Command(
                goto = "chat_node",
                update = {
                    "messages": [ToolMessage(
                        tool_call_id = tool_call_id,
                        content = "Purchase rejected by human reviewer"
                    )]
                }
            )
            # Above we have returned a tool message because for each tool call by the LLM, a tool message is expected in return. An error will be thrown otherwise.
        
    # If no purchase was found, or human said 'yes', proceed to the actual ToolNode
    return Command(goto = 'tools')

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('human_review_node', human_review_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges(
    'chat_node',
    tools_condition,
    {
        "tools": "human_review_node",
        "__end__": END
    }
)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(InMemorySaver())


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "thread-1"}}
    while True:
        user_input = input("You: ").strip()
        if user_input in {"exit", "quit"}:
            print("Bot: Goodbye!")
            break

        input_state = {"messages": [HumanMessage(content = user_input)]}
        response = chatbot.invoke(input_state, config = config)

        # Handle the HITL interrupt if it exists
        while "__interrupt__" in response and response["__interrupt__"]:
            interrupt_message = response["__interrupt__"][0].value
            print(f"{interrupt_message}")
            decision = input("Your decision (yes / no): ")
            response = chatbot.invoke(Command(resume = decision), config = config)
        
        print(f"Bot: {response['messages'][-1].content}\n")
    

