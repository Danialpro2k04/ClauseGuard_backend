import json
import re
import time
import litellm
from litellm import completion

def call_llm(provider: str, model_name: str, api_key: str, system_prompt: str, user_prompt: str) -> dict:
    """Routes the prompt to the specified LLM provider using LiteLLM with explicit rate-limit handling."""
    
    safe_provider = provider.lower()
    formatted_model = f"{safe_provider}/{model_name}" if safe_provider != "openai" else model_name
    
    max_retries = 5
    retry_delay_seconds = 4

    for attempt in range(max_retries):
        try:
            response = completion(
                model=formatted_model,
                api_key=api_key,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                seed=42,
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content
            
            # Strip markdown code fences if present
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_content.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
            
            # Extract object boundaries
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start:end+1]
                
            return json.loads(cleaned)

        except litellm.RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            print(f"⚠️ Groq Rate Limit hit (Attempt {attempt + 1}/{max_retries}). Sleeping {retry_delay_seconds}s...")
            time.sleep(retry_delay_seconds)