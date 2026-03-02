from dotenv import load_dotenv
load_dotenv()
from google import genai
from langsmith import traceable
from google.genai import types
import re
import inspect
import json
#this is used to get metadata of functions

MAX_ITERATIONS = 10
MODEL="gemini-2.5-flash"



def parse_action_input(raw: str) -> dict:
    raw = raw.strip()

    # 1) JSON style: {"price": 60, "discount_tier": "silver"}
    if raw.startswith("{") and raw.endswith("}"):
        return json.loads(raw)

    # 2) key: value pairs (supports commas)
    # e.g. "price: 60, discount_tier: silver"
    # e.g. "price=60, discount_tier=silver"
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    data = {}

    for part in parts:
        if ":" in part:
            k, v = part.split(":", 1)
        elif "=" in part:
            k, v = part.split("=", 1)
        else:
            # fallback: single unnamed value (for single-arg tools)
            data["_pos0"] = part
            continue

        k = k.strip()
        v = v.strip().strip("'\"")
        data[k] = v

    return data







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



#type cast price: as llm inputs float as string (llms only work with strings at base level) -> convert to float for calculation
@traceable(name="Tool")
def apply_discount(price : float, discount_tier: str) -> float:
    """
    This tool is used to apply a discount to a price according to the tier
    available tiers: bronze,silver,gold
    Args:
        price : the price of the product
        discount_tier : the discountt tier
    """
    price=float(price)
    print(f"Applying discount on {price} with {discount_tier} discount")
    discount_percentages={"bronze" : 5, "silver" : 10 , "gold" : 20}
    discount=discount_percentages.get(discount_tier,0)
    return round(price*(1 - discount/100),2)





TOOLS = {"get_price": get_price,
         "apply_discount": apply_discount}


def get_tool_description(TOOL):
    descriptions = []
    for tool_name, tool_function in TOOLS.items():
        # wrapped bypassing to get original function from @traceable
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function)
        descriptions.append(f"{tool_name}: {docstring}")
    return "\n".join(descriptions)



# react prommpt needs tool_descriptions and tool names
tool_description = get_tool_description(TOOLS)
tool_names = ', '.join(TOOLS.keys())

# create the react promt which contains the rules and the OG langchain format
react_prompt = f"""
    STRICT RULES YOU MUST FOLLOW:
    1. Never assume or guess the price of a product, only use prices from the catalog tool.
    2. Never assume or guess the discount, use the discount catalog tool.
    3. Never assume the tier; if not specified, ask the user.
    4. Don't do the math yourself; use apply_discount.
    5. Only apply discount after you get the price using get_price.

    Answer the following questions as best you can. You have access to the following tools:

    {tool_description}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

Question: {{question}}
Thought: """



@traceable(name="Gemini LLM Call", run_type="llm")
def gemini_generate(client, model, config, contents):
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    return resp





#agent loop

@traceable(name="GenAI Agent Loop")
def run_agent(query:str ):

    #create genAi client
    client = genai.Client()
    #config to stop when observation is encountered
    config = types.GenerateContentConfig(
        stop_sequences=["\nObservation"],
        temperature=0
    )
    # contents = [
    #     types.Content(role="user", parts=[types.Part.from_text(text=query)])
    # ]

    print(f"Question: {query}")
    print('='*60)
    # inject question to reAct prompt and intialize scratchpad
    prompt=react_prompt.format(question=query)
    scratchpad=""


    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"--- Iteration: {iteration} ---")

        #append scratchpad
        full_prompt=prompt+scratchpad

        #run llm
        contents=full_prompt
        response = gemini_generate(client, MODEL, config, contents)

        #extract text from response
        response_text=response.text
        print(f"LLM Output: \n{response_text}")


        #Thought -> answer (loop exit condition)
        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", response_text)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer



        #Action loop  (thought -> actiom)
        #parse action_function name and action_inputs
        print(f"  [Parsing] Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", response_text)
        action_input_match = re.search(r"Action Input:\s*(.+)", response_text)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        # Handle key=value OR key: value formats
        tool_name = tool_name.strip()
        tool_input_raw = tool_input_raw.strip()

        if tool_name not in TOOLS:
            observation = f"Error: Tool '{tool_name}' not found."
        else:
            kwargs = parse_action_input(tool_input_raw)

            # If it's a single unnamed value, map it to the tool's first param
            if "_pos0" in kwargs:
                val = kwargs.pop("_pos0")
                # map to correct param name using inspect
                original_fn = getattr(TOOLS[tool_name], "__wrapped__", TOOLS[tool_name])
                first_param = next(iter(inspect.signature(original_fn).parameters))
                kwargs[first_param] = val

            observation = str(TOOLS[tool_name](**kwargs))
            scratchpad += f"{response_text}\nObservation: {observation}\nThought: "

        print(f"  [Tool Result] {observation}")



    # append output to context (tool -> thought)





    print("ERROR: Maxed out iterations")
    return None


if __name__ == "__main__":
    print("Hello from agent!")
    print()
    run_agent("What is the price of a keyboard with silver discount applied?")
