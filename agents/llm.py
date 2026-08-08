import json
import re
from litellm import completion

def call_llm(provider: str, model_name: str, api_key: str, system_prompt: str, user_prompt: str) -> dict:
    """Routes the prompt to the specified LLM provider using LiteLLM."""
    
    # Force the provider string to be lowercase to satisfy LiteLLM's strict formatting
    safe_provider = provider.lower()
    
    # LiteLLM expects format like "groq/gpt-oss-20b" except for OpenAI
    formatted_model = f"{safe_provider}/{model_name}" if safe_provider != "openai" else model_name
    
    response = completion(
        model=formatted_model,
        api_key=api_key,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        seed=42,
        response_format={"type": "json_object"},
        # Added num_retries to automatically wait and retry if Groq rate limits are hit
        num_retries=3
    )
    
    raw_content = response.choices[0].message.content
    
    # Strip markdown code fences if the model included them
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_content.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    
    # Extract object boundaries to drop conversational preamble/postscript
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start:end+1]
        
    return json.loads(cleaned)