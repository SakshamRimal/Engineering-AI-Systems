import json 
import math 

# tool implementation (the actual python code for the tool) goes here

def calculator(expression: str) -> dict:
    """
    Safely evaluate a basic arithmetic expression 
    """
    allowed_name = {"sqrt":math.sqrt, "pow":math.pow, "abs":abs, "round":round}
    try:
        # Evaluate the expression using eval in a restricted environment
        result = eval(expression, {"__builtins__": None}, allowed_name)
        return {"result": result}
    
    except Exception as e:
        return {"error": f"Error evaluating expression: {str(e)}"}
    
def web_search(query: str) -> dict:
    """
    Placeholder web search tool.Replace with a real API (eg. SerpAPI, Bing Search API) for production use.
    Kept as a stub so the tool calling loop works wihtuout extra signups 
    """
    return {
        "results":[
            {"title":f"Stub result for '{query}'", "url":"https://example.com", "snippet":"This is a stub search result. Replace with a real search API."}
        ]
    }
    
def query_knowledge_base(query: str) -> dict:
    """
    Will call into the RAG system to query the knowledge base.
    
    """
    return {
        "chunks": [],
        "note":"RAG retrieval not implemented in this stub. Replace with actual RAG retrieval logic."
    }
    

# tool schemas (what we tell the model is available, and how to call it)

TOOL_DEFINITIONS = [
    {
        "type":"function",
        "function":{
            "name":"calculator",
            "description":"Evaluate a basic arithmetic expression. Allowed functions: sqrt, pow, abs, round. Example: 'sqrt(16) + pow(2,3)'",
            
            "parameters":{
                "type":"object",
                "properties":{
                    "expression":{
                        "type":"string",
                        "description":"The arithmetic expression to evaluate."
                    }
                },
                "required":["expression"]
            },
        },
        
    },
    {
      "type":"function",
      "function" : {
          "name": "query_knowledge_base",
          "description" : "Search the internal document knowledge base for infromation relevant to the user's query. Returns a list of document chunks that may contain the answer.",
          
          "parameters" : {
                "type":"object",
                "properties":{
                    "query":{
                        "type":"string",
                        "description":"The user's query to search for in the knowledge base."
                    }
                },
                "required":["query"]
          },
      },  
    },
]

TOOL_FUNCTIONS = {
    "calculator": calculator,
    "query_knowledge_base": query_knowledge_base,
    "web_search": web_search,
    
}

def execute_tool(name:str , arguments_json: str) -> str:
    """
    Execute a tool by name and return a JSON string result
    """
    if name not in TOOL_FUNCTIONS:
        return json.dumps({"error": f"Tool '{name}' not found"})
    
    try:
        args = json.loads(arguments_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON arguments: {str(e)}"})
    
    result = TOOL_FUNCTIONS[name](**args)
    return json.dumps(result)
    
    
