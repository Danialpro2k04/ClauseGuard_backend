import json
from litellm import completion

def call_llm(provider: str, model_name: str, api_key: str, system_prompt: str, user_prompt: str) -> dict:
    """Routes the prompt to the specified LLM provider using LiteLLM."""
    
    # FIX: Force the provider string to be lowercase to satisfy LiteLLM's strict formatting
    safe_provider = provider.lower()
    
    # LiteLLM expects format like "groq/llama-3.1-8b-instant" except for OpenAI
    formatted_model = f"{safe_provider}/{model_name}" if safe_provider != "openai" else model_name
    
    response = completion(
        model=formatted_model,
        api_key=api_key,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        # Pinned to 0 (rather than 0.1) so identical input clauses produce identical
        # compliance_statement paraphrases and identical risk verdicts across runs.
        # gpt-oss-20b via most providers still has minor residual nondeterminism at
        # temperature 0 (batching/kernel effects), but this removes the dominant
        # source of run-to-run drift.
        temperature=0,
        seed=42,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)