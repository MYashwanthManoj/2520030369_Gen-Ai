import os
from dotenv import load_dotenv

# Step 1: Load API key from .env
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

prompt = "What is Artificial Intelligence? Explain in simple words."

if gemini_key:
    from google import genai

    # Step 2 & 3: Read Gemini API key and initialize client
    client = genai.Client(api_key=gemini_key)

    # Step 4: Make first LLM call
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
    except Exception:
        # Fallback if gemini-3.8-flash experiences temporary high demand (503/504)
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

    # Step 5: Display response
    print("\nLLM Response:")
    print(response.text)

elif openai_key:
    from openai import OpenAI

    client = OpenAI(api_key=openai_key)
    response = client.responses.create(
        model="gpt-5.5",
        input=prompt
    )

    print("\nLLM Response:")
    print(response.output_text)

else:
    raise ValueError("Gemini API key not found! Please check your .env file.")
