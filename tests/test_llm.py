from app.llm_client import llm_client
from app.schemas import AssistantAnswer

if __name__ == "__main__":
    result = llm_client.chat_structured(
        "What is the capital of France?"
    )
    print("Raw dict:", result)

    validated = AssistantAnswer(**result)
    print("Validated object:", validated)