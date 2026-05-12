import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

from google import genai

client = genai.Client(api_key=api_key)

gen_res = client.models.generate_content(model="gemma-4-31b-it",contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.")

print(gen_res.text)


metadata = gen_res.usage_metadata

print(f"Prompt tokens: {metadata.prompt_token_count}\nResponse tokens: {metadata.candidates_token_count}")












