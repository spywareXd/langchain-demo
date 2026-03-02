from dotenv import load_dotenv
load_dotenv()
from google import genai
from langsmith import traceable
from google.genai import types

MAX_ITERATIONS = 10
MODEL="gemini-2.5-flash"
SYSTEM_PROMPT = (
    "You are a helpful shopping assistant.\n"
    "You help users calculate the final price after discount.\n"
    "You have access to product catalog tool.\n"
    "You have access to the discount catalog tool.\n"
    "STRICT RULES YOU MUST FOLLOW:\n"
    "1. Never assume or guess the price of a product, only use prices from the catalog tool.\n"
    "2. Never assume or guess the discount, use the discount catalog tool.\n"
    "3. Never assume the tier; if not specified, ask the user.\n"
    "4. Don't do the math yourself; use apply_discount.\n"
    "5. Only apply discount after you get the price using get_price.\n"
)



# declare get_price function
get_product_declaration = {
    "name": "get_price",
    "description": "Gets the base price of a product",
    "parameters": {
        "type": "object",
        "properties": {
            "product": {
                "type": "string",
                "description": "The product that the user wants to find the price about after applying discount",
            },
        },
        "required": ["product"],
    },
}

apply_discount_declaration = {
    "name": "apply_discount",
    "description": "Applies the discount to the price of a product",
    "parameters": {
        "type": "object",
        "properties": {
            "price": {
                "type": "number",
                "description": "The price of the product that the user wants to find the price about after applying discount from the catalog provided in the function",
            },
            "discount_tier": {
                "type": "string",
                "description": "The discount_tier to be applied : gold/silver/bronze to the product from the catalog in the function",
                "enum": ["gold","silver","bronze"],  #enum makes it so the valye MUST be from this list
            },

        },
        "required": ["price", "discount_tier"],
    },
}





@traceable(name="Tool")
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

@traceable(name="Tool")
def apply_discount(price : float, discount_tier: str) -> float:
    """
    This tool is used to apply a discount to a price according to the tier
    available tiers: bronze,silver,gold
    Args:
        price : the price of the product
        discount_tier : the
    """
    print(f"Applying discount on {price} with {discount_tier} discount")
    discount_percentages={"bronze" : 5, "silver" : 10 , "gold" : 20}
    discount=discount_percentages.get(discount_tier,0)
    return round(price*(1 - discount/100),2)


@traceable(name="Gemini LLM Call", run_type="llm")
def gemini_generate(client, model, contents, config):
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    return resp





#agent loop

@traceable(name="GenAI Agent Loop")
def run_agent(query:str ):
    client = genai.Client()
    tools_decl = types.Tool(function_declarations=[get_product_declaration, apply_discount_declaration])
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[tools_decl],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    #automatic_function_calling -> The llm will not run the function automatically. It will only pass the decision, we will run the tool


    tools=[get_price,apply_discount]
    tools_dict={
        "get_price":get_price,
        "apply_discount":apply_discount,
    }   #explicity define the dictionary as t.name will not work as there is no name field created due to lack of tool decorator

    # messages replaced by contents
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=query)])
    ]

    print(f"Question: {query}")
    print('='*60)

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"--- Iteration: {iteration} ---")
        #run llm
        response = gemini_generate(client, MODEL, contents, config)

        #tool_call field replaced by function_call
        # If model is done, it will return no function_calls and you can use response.text
        function_calls = response.function_calls or []
        if not function_calls:
            print(f"Final Answer: {response.text}")
            return response.text


        #LLM may request more than one Tool call at once
        #FORCE ONLY ONE TOOL PER ITERATION:\
        #If LLM request tool_calls= [func1, func 2] -> only run tool_calls[0]

        tool_call = function_calls[0]
        tool_name=tool_call.name
        tool_args = dict(tool_call.args)
        #the args that the llm wants to pass into the tool function

        print(f"Tool selected: {tool_name} with args: {tool_args} ")

        tool_to_use=tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Error-> Tool {tool_name} not found")

        # Execute tool
        observation = tool_to_use(**tool_args)  #kwaargs unpacking
        print(f"Tool Result: {observation}")

        #append output to context
        function_call_content = response.candidates[0].content
        contents.append(function_call_content)
        # Append tool response as a "tool" role message
        function_response_part = types.Part.from_function_response(
            name=tool_name,
            response={"result": observation},
        )
        contents.append(types.Content(role="tool", parts=[function_response_part]))


    print("ERROR: Maxed out iterations")
    return None


if __name__ == "__main__":
    print("Hello from agent!")
    print()
    run_agent("What is the price of a keyboard with silver discount applied?")
