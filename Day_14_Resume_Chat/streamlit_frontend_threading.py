import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
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
    try:
        return chatbot.get_state(config = {"configurable": {"thread_id": thread_id}}).values['messages']
    except KeyError:
        # thread_id doesn't have any conversation
        raise

# **************************************** Session Setup ******************************
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

add_thread(st.session_state['thread_id'])

# **************************************** Sidebar UI *********************************
st.sidebar.title("Langgraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        try:
            messages = load_conversations(thread_id)
        except KeyError:
            # This thread ID doesn't have any conversation
            st.session_state['message_history'] = []
            pass
        else:
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

CONFIG = {"configurable": {"thread_id": st.session_state['thread_id']}}

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content = user_input)]},
                config = CONFIG,
                stream_mode = "messages"
            )
        )
    
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})