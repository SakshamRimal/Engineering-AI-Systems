from openai import OpenAI
from app.config import settings
from typing import Any, cast

from app.tools import TOOL_DEFINITIONS, execute_tool

# System prompt: defines role, boundaries, and when to lean on retrieved context
SYSTEM_PROMPT = """You are a helpful, factual AI assistant.

Rules you must follow:
1. If the user's question relates to retrieved context provided to you, base your answer 
   primarily on that context. Do not use outside knowledge to contradict it.
2. If no relevant context is provided and you don't know the answer, say so honestly 
   instead of guessing.
3. When you use a tool, wait for its result before answering — never fabricate a tool result.
4. Keep answers concise and directly address the question asked.
5. If asked to produce structured data, respond with valid JSON only, matching the 
   requested schema exactly — no extra commentary, no markdown code fences.
"""

class LLMClient:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER

        if self.provider == "openai":
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.model = settings.OPENAI_MODEL
        elif self.provider == "vllm":
            # vLLM exposes an OpenAI-compatible server, so we reuse the same client class
            self.client = OpenAI(base_url=settings.VLLM_BASE_URL, api_key="not-needed")
            self.model = settings.VLLM_MODEL
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

    def chat(
        self,
        user_message: str,
        history: list[dict] | None = None,
        temperature: float = 0.2,
        top_p: float = 1.0,
        max_tokens: int = 800,
        system_prompt: str | None = None,
    ) -> str:
        """
        Basic chat call — no tools, no JSON enforcement.
        history: list of {"role": "user"|"assistant", "content": str}
        """
        messages = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    
    def chat_structured(
        self , 
        user_message: str,
        context: str = "",
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> dict:
        """
        Forces valid JSON outpout matchin the AssistantAnswer schema. If the model fails to produce valid JSON, it will retry a few times.
        """
        
        schema_instructions = """
        Respond with ONLY a JSON object in exactly this shape, no other text:
        {
        "answer": "<your answer as a string>",
        "sources": [{"document": "<filename>", "chunk_id": "<id>"}],
        "confidence": <float between 0 and 1>
        }
        If you used no retrieved context, return an empty "sources" array.
        """
        
        full_system = SYSTEM_PROMPT + schema_instructions
        
        user_content = user_message
        if context:
            user_content = f"Context:\n{context}\n\nQuestion:\n{user_message}"
        
        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_content}
        ]
        
        #OPENAI JSON mode
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type":"json_object"},
        )
        raw = response.choices[0].message.content
        return self._parse_json_with_repair(raw, messages)
    
    def chat_with_tools(
        self,
        user_message: str,
        history: list[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        max_iterations: int = 5,
    ) -> dict:
        """
        Runs the tool-calling loop: model may request tool calls, we execute them,
        feed results back, repeat until the model returns a final text answer.
        Returns: {"answer": str, "tool_calls_made": [str, ...]}
        """
        from app.tools import TOOL_DEFINITIONS, execute_tool

        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        tool_calls_made = []

        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
            )
            choice = response.choices[0]
            message = choice.message

            # No tool call requested -> model has its final answer
            if not message.tool_calls:
                return {"answer": message.content, "tool_calls_made": tool_calls_made}

            # Append the assistant's tool-call request to the conversation
            function_tool_calls = [cast(Any, tc) for tc in message.tool_calls]
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in function_tool_calls
                ],
            })

            # Execute each requested tool call and append its result
            for tc in function_tool_calls:
                tool_calls_made.append(tc.function.name)
                result = execute_tool(tc.function.name, tc.function.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        # Safety cap hit — force a final answer without further tool calls
        final = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {"answer": final.choices[0].message.content, "tool_calls_made": tool_calls_made}
        
    
    def _parse_json_with_repair(self, raw: str, original_messages: list[dict]) -> dict:
        import json

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Repair attempt: ask the model to fix its own broken output
            repair_messages = original_messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "That was not valid JSON. Return ONLY the corrected valid JSON object, nothing else."},
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=repair_messages,
                temperature=0,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            raw_retry = response.choices[0].message.content
            return json.loads(raw_retry)  # if this still fail


# Singleton instance used across the app
llm_client = LLMClient()