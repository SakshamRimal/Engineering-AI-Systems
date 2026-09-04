from app.llm_client import llm_client

if __name__ == "__main__":
    reply = llm_client.chat("In one sentence, what is Retrieval-Augmented Generation?")
    print(reply)