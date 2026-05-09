import os
from dotenv import load_dotenv

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.llms import ChatMessage

load_dotenv()


def load_gemini_model():
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key= gemini_api_key
    )

    return llm

