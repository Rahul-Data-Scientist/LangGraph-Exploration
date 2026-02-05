from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Annotated, TypedDict
import requests
import os


load_dotenv()

stock_api_key = os.environ["STOCK_API_KEY"]

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
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """

    decision = interrupt(f"Shall I go ahead with order of {quantity} stocks of {symbol}?")

    if isinstance(decision, str) and decision.strip().lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} stocks of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} stocks of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }


tools = [get_stock_price, purchase_stock]
llm = ChatOpenAI(model = "gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state['messages'])
    return {"messages": [response]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node", END)

checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer = checkpointer)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-thread"}}
    while True:
        user_input = input("You: ")
        if user_input in {"exit", "quit"}:
            print("Bot: Goodbye!")
            break
        
        input_state = {"messages": [HumanMessage(content = user_input)]}
        response = chatbot.invoke(input_state, config = config)

        interrupts = response.get("__interrupt__", [])

        if interrupts:
            message = interrupts[0].value
            print(message)
            descision = input("Your decision (yes / no): ")
            response = chatbot.invoke(Command(resume = descision), config = config)
        
        print("Bot:", response["messages"][-1].content)
        print()
