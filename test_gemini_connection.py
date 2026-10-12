import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError


MODEL = "gemini-3.8-flash"
PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        print("Authentication succeeded: no")
        print(f"Model used: {MODEL}")
        print("Response received: no")
        print("Error: GEMINI_API_KEY is missing from the environment and project .env file.")
        return 1

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=MODEL,
            contents="Reply with exactly: Gemini connection successful.",
        )
    except (APIError, httpx.HTTPError) as error:
        message = str(error).replace(api_key, "[REDACTED]")
        print("Authentication succeeded: no")
        print(f"Model used: {MODEL}")
        print("Response received: no")
        print(f"Error: {message}")
        return 1

    print("Authentication succeeded: yes")
    print(f"Model used: {MODEL}")
    print(f"Response received: {'yes' if response is not None else 'no'}")
    return 0 if response is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
