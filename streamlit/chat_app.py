import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="LangChain Chat App", page_icon="🦜")

st. title("🦜 LangChain Chat App")

st.write("Welcome to the LangChain Chat App! 😎")

models = ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]

with st.expander("Learn more about models and temperature"):
    cols1 = st.columns(2)
    with cols1[0]:
        st.markdown("""
        ### Models
        - **llama3-70b-8192**: Largest model, best performance, slower
        - **llama3-8b-8192**: Smaller model, faster, less capable
        - **mixtral-8x7b-32768**: Mixture of experts model, good balance of performance and speed
        - **gemma-7b-it**: Google's Gemma model, instruction-tuned version""")
    with cols1[1]:
        st.markdown("""
        ### Temperature
        Temperature affects the randomness of the model's output:
        - **0.0**: Most deterministic, always choosing the most likely next token
        - **0.5**: Balanced between deterministic and random
        - **1.0**: Standard setting, introduces some variability
        - **2.0**: Most random, can lead to more creative but potentially less coherent responses
        
        Choose a lower temperature for fact-based tasks and a higher temperature for creative tasks.
        """)
cols = st.columns(2)
with cols[0]:
    model = st.selectbox(label="**LLM Model**", 
                    options=models,
                    index=2
    )
with cols[1]:
    temperature = st.slider(label="**Temperature**",
                        min_value=0.0,
                        max_value=2.0,
                        value=1.0,
                        step=0.01
    )
clicked = st.button("Clear History")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant pretending to be Eminem"),
        ("placeholder", "{chat_history}"),
        ("user", "{query}"),
    ]
)

llm = ChatGroq(model=model, temperature=temperature)

chain = prompt | llm | StrOutputParser()

msgs = StreamlitChatMessageHistory()
if len(msgs.messages) == 0 or clicked:
    msgs.clear()
    msgs.add_ai_message(chain.invoke(
        {"query": "Write a short welcome message",
         "chat_history": []}
    ))

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="query",
    history_messages_key="chat_history",
)

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)


chat_input = st.chat_input("Ask me anything!")

if chat_input:
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"query": chat_input}, config)
    st.chat_message("user").write(chat_input)
    st.chat_message("ai").write(response)
    with st.expander("view chat history"):
        for i, msg in enumerate(msgs.messages):
            st.write(f"{i}. {msg.type.upper()}: {msg.content}")
    st.write(llm.model_name)
