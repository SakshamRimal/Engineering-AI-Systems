from app.llm_client import llm_client

if __name__ == "__main__":
    result = llm_client.chat_with_tools("What is 15% of 340, and then add 7 to that?")
    print("Answer:", result["answer"])
    print("Tools used:", result["tool_calls_made"])