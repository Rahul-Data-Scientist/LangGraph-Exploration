import streamlit as st
from langgraph_tool_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# **************************************** utility functions *************************
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []

def load_conversations(thread_id):
    state = chatbot.get_state(config = {"configurable": {"thread_id": thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])

# **************************************** Session Setup ******************************
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])

# **************************************** Sidebar UI *********************************
st.sidebar.title("Langgraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversations(thread_id)
        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        
        st.session_state['message_history'] = temp_messages



# **************************************** Main UI ************************************
# Load the conversation history
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("Type here...")

# This is the only change from previous day's streamlit frontend
CONFIG = {
    "configurable": {"thread_id": st.session_state['thread_id']},
    "metadata": {"thread_id": st.session_state['thread_id']},
    "run_name": "chat_turn"
}

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content = user_input)]},
                config = CONFIG,
                stream_mode = "messages"
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, 'name', 'tool')

                    if status_holder['box'] is None:
                        status_holder['box'] = st.status(
                            label = f"🔧 Using `{tool_name}` …",
                            state = "running",
                            expanded = True
                        )
                    else:
                        status_holder['box'].update(
                            label = f"🔧 Using `{tool_name}` …",
                            state = "running",
                            expanded = True
                        )

                elif isinstance(message_chunk, AIMessage):
                    yield message_chunk.content
        
        ai_message = st.write_stream(ai_only_stream())

        if status_holder['box'] is not None:
            status_holder['box'].update(
                label = "✅ Tool finished",
                state = "complete",
                expanded = False
            )
    
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

    