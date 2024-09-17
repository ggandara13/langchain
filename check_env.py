import os

openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    print(f"OpenAI API key found: {openai_api_key[:5]}...")
else:
    print("ALERT: OpenAI API key is not set.")
