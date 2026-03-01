from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain.tools import tool
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL="gemini-2.5-flash"

@tool
def get_price(product: str) -> float:
    """
    This tool is used to look up the price of a product
    Args:
        product : the name of the product
    Returns:
        the price of the product
    """
    print(f"Getting price of {product}")
    products={"laptop" : 1500, "mouse" : 20, "keyboard" : 60}
    return products.get(product,0)
#dict.get(x,y) -> if key x exists return the value else return default value y

@tool
def apply_discount(price : float, discount_tier: str) -> float:
    """
    This tool is used to apply a discount to a price according to the tier
    available tiers: bronze,silver,gold
    Args:
        price : the price of the product
        discount_tier : the
    """
    print(f"Applying discount on {price} with {discount_tier} discount")
    discount_precentages={"bronze" : 5, "silver" : 10 , "gold" : 20}
    discount=discount_precentages.get(discount_tier,0)
    return round(price*(1 - discount/100),2)


#agent loop

@traceable(name="LangChain Agent Loop")
def run_agent(query:str ):
    tools=[get_price,apply_discount]
    tools_dict={t.name : t for t in tools}
    llm=init_chat_model(f"google_genai:{MODEL}")
    llm_with_tools=llm.bind_tools(tools)
    print(f"Question: {query}")
    print('='*60)

    messages=[
        SystemMessage(content=
                        "You are a helpful shopping assiatant\n"
                      "You help users to calculate the price of a product after discount\n"
                      "You have access to product catalog tool\n"
                      "You have access to the discount catalog tool \n"
                        #DEFENSIVE PROMPTING TO AVOID HALLUCINATION:
                      "STRICT RULES YOU MUST FOLLOW\n  "
                      "1. Never assume or guess the price of a product, only use prices for the catalog provided in the tools\n"
                      "2. Never assume or guess the discount to be applied, use the discount catalog in the tool\n"
                      "3. Never assume or guess what tier discount to apply, if no tier is specified, ask the user for it\n"
                      "4. Dont try to calculate the discount yourself using math, refer to the apply_discount function\n"
                      "5. Only apply discount using apply_discount function after you get the price of the product using the get_price function\n"),

        HumanMessage(content=query)

    ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"--- Iteration: {iteration} ---")
        ai_message=llm_with_tools.invoke(messages)
        #ai_message will have tool_call field if it requires a tool_calls otherwise itll have content-> task done exit loop

        tool_calls=ai_message.tool_calls

        if not tool_calls:
            print(f"Final Answer:  {ai_message.content}")
            return ai_message.content

        #LLM may request more than one Tool call at once
        #FORCE ONLY ONE TOOL PER ITERATION:\
        #If LLM request tool_calls= [func1, func 2] -> only run tool_calls[0]

        tool_call=tool_calls[0]
        tool_name=tool_call.get("name")
        tool_args=tool_call.get("args", {})
        #the args that the llm wants to pass into the tool function
        tool_id=tool_call.get("id")

        print(f"Tool selected: {tool_name} with args: {tool_args} ")
        tool_to_use=tools_dict.get(tool_name,None)  #tool_to_use becomes a tool object that can be invoked

        observation=tool_to_use.invoke(tool_args)
        if tool_to_use is None:
            raise ValueError(f"Error-> Tool {tool_name} not found")

        print(f"Tool Result: {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_id))

    print("ERROR: Maxed out iterations")
    return None


if __name__ == "__main__":
    print("Hello from agent!")
    print()
    result=run_agent("What is the price of a keyboard with silver discount applied?")
