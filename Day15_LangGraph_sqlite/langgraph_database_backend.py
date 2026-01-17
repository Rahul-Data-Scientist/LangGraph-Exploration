from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from datetime import datetime

load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOpenAI()

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

conn = sqlite3.connect('chatbot.db', check_same_thread = False)
# We did check_same_thread = False because we will have multiple threads but default value of check_same_thread is True, due to which our sqlite databse can track only one thread

checkpointer = SqliteSaver(conn = conn)

chatbot = graph.compile(checkpointer = checkpointer)

# def retrieve_all_threads():
#     all_threads = set()
#     for checkpoint in checkpointer.list(None):
#         # checkpointer.list returns either of these two:
#         # 1. All the checkpoints saved in the database -> which we wanted, that's why we passed None under checkpointer.list()
#         # 2. All the checkpoints saved in the database for a particular thread -> IN this case, we have to pass the thread under checkpointer.list()

#         all_threads.add(checkpoint.config['configurable']['thread_id'])
    
#     return list(all_threads)

# Above function works correctly but we want our function to return the threads in increasing order of their creation time i.e., latest thread comes last in the list whereas oldest thread comes first in the list.

def retrieve_all_threads():
    threads_dict = {}

    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        thread_ts = checkpoint.checkpoint['ts'] # # ISO timestamp string
        thread_ts = datetime.fromisoformat(thread_ts.replace('Z', '+00:00'))
        # we are doing thread_ts.replace('Z', '+00:00') because:
        # LangGraph timestamps may end with "Z" (UTC), which datetime.fromisoformat() does not accept.
        # So we replace "Z" with "+00:00" to make the timestamp Python-compatible.
        # Then we convert the ISO timestamp string into a datetime object for correct time comparison and sorting.

        if thread_id not in threads_dict:
            threads_dict[thread_id] = thread_ts
        else:
            threads_dict[thread_id] = min(threads_dict[thread_id], thread_ts)
    
    sorted_threads = sorted(threads_dict.items(), key = lambda x: x[1])
    return [thread_id for thread_id, _ in sorted_threads]
