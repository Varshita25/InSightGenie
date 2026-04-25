import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
# Correct logic: Try to get from environment first, then fallback to your hardcoded key
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    # Use the key you provided directly
    # Use the new key you provided
    api_key = "AIzaSyDfitBn1Nyr--00-rq_tz_VRQM6uhJk4Yg"

genai.configure(api_key=api_key)

def ask_gemini(prompt: str, model="models/gemini-flash-latest", temperature=0.3):
    """
    Send prompt to Gemini with a fallback if quota is exceeded.
    """
    # Try models in order of speed/quota availability
    fallback_models = ["models/gemini-flash-latest", "models/gemini-1.5-flash", "models/gemini-pro-latest"]
    
    last_error = ""
    for m in fallback_models:
        try:
            model_instance = genai.GenerativeModel(m)
            response = model_instance.generate_content(
                prompt,
                generation_config={"temperature": temperature}
            )
            return response.text
        except Exception as e:
            last_error = str(e)
            if "429" in last_error or "quota" in last_error.lower():
                continue # Try next model
            break # Non-quota error, stop trying
            
    return f"AI is temporarily busy (Quota reached). Please try again in 30 seconds. (Error: {last_error})"
