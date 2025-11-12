import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=api_key)

def ask_gemini(prompt: str, model="gemini-1.5-flash", temperature=0.3):
    """
    Send prompt to Gemini and return response.
    """
    try:
        response = genai.GenerativeModel(model).generate_content(
            prompt,
            generation_config={"temperature": temperature}
        )
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API Error: {str(e)}"
