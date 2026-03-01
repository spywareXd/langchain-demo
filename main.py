import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from typing import List
from pydantic import BaseModel, Field


tavily=TavilyClient()

#pydantic:
class Source(BaseModel):
    """Schema for source URL used by agent"""
    url:str = Field(description="URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response"""
    answer:str = Field(description="Answer of the agent")
    sources: List[Source] = Field(default_factory=list, description="Sources of the agent")

#@tool
# def search(query : str) -> str:
#     """
#     A tool that is used to search of the internet
#     Args:
#         query: string to be searched on the internet
#     Returns:
#         The search result
#     """
#
#     print(f"LLM searching for {query}")
#     return tavily.search(query=query)




tools=[TavilySearch()]

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
agent=create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result=agent.invoke( {"messages" : [HumanMessage(content="Can you list some job postings for a 3D blender artist in delhi, bangalore or any other city in India? Also tell me what degree to pursure to land such a job.")] } )
    print(result)

if __name__ == "__main__":
    main()
