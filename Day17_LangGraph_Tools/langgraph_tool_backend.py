from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
from datetime import datetime
import sqlite3
import requests
import os

load_dotenv()

"""
ToolNode
In LangGraph, a ToolNode is a prebuilt node type that acts as a bridge between your graph and external tools (functions, APIs, utilities).

Normally in LangGraph you'd write a node function yourself: it takes in state and returns state.

A ToolNode is a ready-made node that knows how to handle a list of LangChain tools.

Its job: listen for tool calls from the LLM (like "call search()" or "get_weather()") and automatically route the request to the correct tool, then pass the tool's output back into the graph.


tools_condition
tools_condition is a prebuilt conditional edge function that helps your graph decide:

"Should the flow go to the ToolNode next, or back to the LLM?"
"""

stock_api_key = os.environ["STOCK_API_KEY"]

llm = ChatOpenAI(model = "gpt-4.1-mini")

# Tools

search_tool = DuckDuckGoSearchRun(region = "us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """

    # It is recommended to add docstring to each custom tool so that our LLM will understand the function of each tool.

    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stock_api_key}"
    r = requests.get(url)
    return r.json()

tools = [search_tool, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tools = tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
     """LLM node that may answer or request a tool call."""
     messages = state['messages']
     response = llm_with_tools.invoke(messages)
     return {"messages": [response]}

tool_node = ToolNode(tools)

conn = sqlite3.connect(database = "chatbot.db", check_same_thread = False)
checkpointer = SqliteSaver(conn = conn)

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer = checkpointer)

def retrieve_all_threads():
    threads_dict = {}

    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        thread_ts = checkpoint.checkpoint['ts'] # # ISO timestamp string
        thread_ts = datetime.fromisoformat(thread_ts.replace('Z', '+00:00'))

        if thread_id not in threads_dict:
            threads_dict[thread_id] = thread_ts
        else:
            threads_dict[thread_id] = min(threads_dict[thread_id], thread_ts)
    
    sorted_threads = sorted(threads_dict.items(), key = lambda x: x[1])
    return [thread_id for thread_id, _ in sorted_threads]