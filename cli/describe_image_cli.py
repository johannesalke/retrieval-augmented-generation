import argparse
import hybrid_search as hs
from lib.semantic_search import load_movies
import lib.llm_functions as llm
import mimetypes
import os
from dotenv import load_dotenv
from google import genai
import types

def main():
    parser = argparse.ArgumentParser(description="Image Description CLI")
    #subparsers = parser.add_subparsers(dest="command", help="Available commands")
    parser.add_argument("--query", type=str, help="Query to be sought for")
    parser.add_argument("--image", type=str, help="Image address")

    
    args = parser.parse_args()
    if not args.query or not args.image:
        raise Exception("Missing argument: Both query and image reference required")


    
    

    if not os.path.exists(args.image):
        raise Exception("Image does not exist")
    mime,_ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image,"rb") as f:
        image = f.read()
    


    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    prompt =  f"""Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""
    client = genai.Client(api_key=api_key)

    parts = [
        prompt,
        genai.types.Part.from_bytes(data=image, mime_type=mime),
        args.query.strip()
    ]

    response = client.models.generate_content(model="gemma-4-31b-it",contents=parts)
    
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()