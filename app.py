import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper,DuckDuckGoSearchAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,ARXIV_MAX_QUERY_LENGTH=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

wikipedia_wrapper=WikipediaAPIWrapper(top_k_results=1,ARXIV_MAX_QUERY_LENGTH=200)
wikipedia=WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

duckduck=DuckDuckGoSearchAPIWrapper()
duckduckgo=DuckDuckGoSearchRun(api_wrapper=duckduck)

st.title("🔎 Search engine chatbot")

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")


if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(api_key=api_key,model="Llama3-8b-8192")
    tools=[duckduckgo,wikipedia,arxiv]

    search_agent=initialize_agent(tools=tools,llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container())
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)


# Display footer for additional info
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Renga Rajan K")
st.sidebar.markdown("Powered by Streamlit")