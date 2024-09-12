import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper,DuckDuckGoSearchAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

llm=ChatGroq(api_key=GROQ_API_KEY,model="Llama3-8b-8192")

arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,ARXIV_MAX_QUERY_LENGTH=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

wikipedia_wrapper=WikipediaAPIWrapper(top_k_results=1,ARXIV_MAX_QUERY_LENGTH=200)
wikipedia=WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

duckduck=DuckDuckGoSearchAPIWrapper()
duckduckgo=DuckDuckGoSearchRun(api_wrapper=duckduck)

tools=[duckduckgo,wikipedia,arxiv]

def generate_response(query_text):
    search_agent=initialize_agent(tools=tools,llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)
    response=search_agent.run(query_text)
    return response

if __name__=="__main__":
    query_text="What is the current state of quantum computing?"
    response=generate_response(query_text=query_text)
    print(response)