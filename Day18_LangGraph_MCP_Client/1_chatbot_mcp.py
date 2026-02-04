# Since here we are making a LangGraph MCP client which connects to MCP server made by FastMCP, we are going to use asyncio as FastMCP server works only with asyncio environment.

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages   # We are using it although iss code mein iska koi use nahi hai as we are not implementing persistence here.
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import os

load_dotenv()

CLOUD_API_KEY = os.environ["FASTMCP_BEARER_TOKEN"]

SERVERS = { 
    "expense": {
        "transport": "streamable_http",  # if this fails, try "sse"
        "url": "https://stormy-aquamarine-loon.fastmcp.app/mcp",
        "headers": {
            "Authorization": f"Bearer {CLOUD_API_KEY}"
        }
    },
    "math": {
        "transport": "stdio",
        "command": "C:\\Python 3114\\Scripts\\uv.exe",
        "args": [
            "run",
            "fastmcp",
            "run",
            "main.py"
        ],
        "cwd": "C:\\Users\\Administrator\\Desktop\\MCP Tutorial\\mcp-math-server"
    }
}

llm = ChatOpenAI(model = 'gpt-4o-mini')
client = MultiServerMCPClient(SERVERS)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def build_graph():
    tools = await client.get_tools()

    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState):
        messages = state['messages']
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}
    
    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)

    graph.add_node('chat_node', chat_node)
    graph.add_node('tools', tool_node)

    graph.add_edge(START, 'chat_node')
    graph.add_conditional_edges('chat_node', tools_condition)
    graph.add_edge('tools', 'chat_node')
    graph.add_edge('chat_node', END)

    chatbot = graph.compile()

    return chatbot

async def main():
    chatbot = await build_graph()
    result = await chatbot.ainvoke({"messages": HumanMessage(content = "Using the provided tool, get me the sum of 500 and 200")})
    print(result['messages'][-1].content)

if __name__ == "__main__":
    asyncio.run(main())