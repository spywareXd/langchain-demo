import os
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
tavily=TavilyClient()

@tool
def search(query : str) -> str:
    """
    A tool that is used to search of the internet
    Args:
        query: string to be searched on the internet
    Returns:
        The search result
    """

    print(f"LLM searching for {query}")
    return tavily.search(query=query)

tools=[search]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)
agent=create_agent(model=llm, tools=tools)






def main():
    print("Hello from langchain-course!")
    result=agent.invoke( {"messages" : [HumanMessage(content="Can you list some job postings for a 3D blender artist in delhi, bangalore or any other city in India? Also tell me what degree to pursure to land such a job.")] } )
    print(result)

if __name__ == "__main__":
    main()
