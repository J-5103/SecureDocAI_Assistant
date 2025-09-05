# src/services/gemini_service.py
from google import genai
import os

def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    # genai.Client() env se key pick karta hai, par explicit bhi de rahe
    return genai.Client(api_key=api_key) if api_key else genai.Client()
