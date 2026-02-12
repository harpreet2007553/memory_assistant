import google.generativeai as genai
import io
from PIL import Image
from dotenv import load_dotenv
from memory_assistant.retrieval import retrieve_multimodal
import os
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
load_dotenv()
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# Convert PIL image to bytes
img_bytes , text = retrieve_multimodal("red car with four wheels")

def relate(img_bytes, text):
    response = model.generate_content([
    {"role": "user", "parts": [
        {"text": f"Is this image related to {text}? Answer Yes or No."},
        {"inline_data": {"mime_type": "image/png", "data": img_bytes}}
    ]}
    ])

    return response.text


print(relate(img_bytes, text))

