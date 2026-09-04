from app.llm_client import llm_client

if __name__ == "__main__":
    result = llm_client.chat_with_tools(
        "Based on the knowledge base, what is decorator and generator in python? Please provide a brief explanation and an example for each."
    )
    print("Answer:", result["answer"])
    print("Tools used:", result["tool_calls_made"])