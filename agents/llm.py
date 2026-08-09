import json
import re
import time
import litellm
from litellm import completion

def call_llm(provider: str, model_name: str, api_key: str, system_prompt: str, user_prompt: str) -> dict:
    """Routes the prompt to the specified LLM provider using LiteLLM."""
    
    safe_provider = provider.lower()
    formatted_model = f"{safe_provider}/{model_name}" if safe_provider != "openai" else model_name
    
    # Append a strict instruction to the system prompt to enforce JSON output
    strict_system_prompt = system_prompt + "\n\nIMPORTANT: You must respond ONLY with a valid JSON object. Do not include any conversational filler."

    max_retries = 5
    retry_delay_seconds = 7  # Keep the delay to respect Groq rate limits

    for attempt in range(max_retries):
        try:
            response = completion(
                model=formatted_model,
                api_key=api_key,
                messages=[
                    {"role": "system", "content": strict_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                seed=42
                # REMOVED: response_format={"type": "json_object"} to stop Groq from crashing on minor formatting hiccups
            )
            
            raw_content = response.choices[0].message.content
            
            # Strip markdown code fences if present
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_content.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
            
            # Extract object boundaries to drop any conversational preamble
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
            
        except json.JSONDecodeError as e:
            # If the model fails to output valid JSON, catch it and retry instead of crashing
            if attempt == max_retries - 1:
                print(f"Failed to parse JSON after {max_retries} attempts. Raw output: {raw_content}")
                raise e
            print(f"⚠️ Invalid JSON received. Retrying (Attempt {attempt + 1}/{max_retries})...")
            time.sleep(2)