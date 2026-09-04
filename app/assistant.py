from app.config import settings
from app.llm_client import get_llm_client


client = get_llm_client()


def ask_ai(message: str):

    response = client.chat.completions.create(
        model=(
            settings.OPENAI_MODEL
            if settings.LLM_PROVIDER == "openai"
            else settings.VLLM_MODEL
        ),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant for a project "
                    "management system. Give clear and concise answers."
                )
            },
            {
                "role": "user",
                "content": message
            }
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content