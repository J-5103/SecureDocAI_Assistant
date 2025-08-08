import base64
import requests

class ImageQuestionPipeline:
    def __init__(self, ollama_url="http://192.168.0.88:11434/api/generate"):
        self.ollama_url = ollama_url
        self.model_name = "llava:13b"

    def run(self, front_image_bytes, back_image_bytes=None):
        front_base64 = base64.b64encode(front_image_bytes).decode("utf-8")
        back_base64 = base64.b64encode(back_image_bytes).decode("utf-8") if back_image_bytes else None

        messages = [
            {
                "role": "system",
                "content": """
You are an AI contact extraction agent. Your job is to analyze one or two images of a business card and extract all visible contact details without guessing. You must return the data only in the three requested formats below.
"""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
You are given one or two visiting card images (front and optionally back).

Extract and return all available contact information from the card. Then return the data in exactly **three formats**:

---

📲 *1. WhatsApp Message:*  
- Use emojis and clean markdown formatting  
- Include only available details  
- For social links, use:  
  - 🔗 YouTube → [Channel Name](url)  
  - 📸 Instagram → [@handle](url)  
  - 👍 Facebook → [Page Name](url)  
  - 💼 LinkedIn → [Profile Name](url)  
  - 📌 Pinterest → [Board Name](url)  
- If a QR code points to location, add: 📍 Scan QR for location

---

📇 *2. vCard (.vcf format):*  
- Use vCard version 3.0  
- Include only present fields  
- Use international phone format  
- Add social media links using `X-SOCIALPROFILE`  
- Add full structured address if available

---

📦 *3. JSON:*  
Return a valid JSON object with the following structure. Omit fields that are not found on the card:

{
  "name": "",
  "designation": "",
  "company": "",
  "phone_numbers": [],
  "emails": [],
  "website": "",
  "address": {
    "line1": "",
    "district": "",
    "city": "",
    "state": "",
    "postal_code": "",
    "country": ""
  },
  "social_media": {
    "instagram": "",
    "facebook": "",
    "youtube": "",
    "linkedin": "",
    "pinterest": ""
  }
}

---

⚠️ RULES:  
- No placeholder or empty fields  
- Only return these 3 output sections: WhatsApp message, vCard, JSON  
- Do not add extra explanation, headings, or commentary
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{front_base64}"
                        }
                    }
                ] + (
                    [{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{back_base64}"
                        }
                    }] if back_base64 else []
                )
            }
        ]

        response = requests.post(self.ollama_url, json={
            "model": self.model_name,
            "messages": messages,
            "stream": False
        })

        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "")
        else:
            return f"❌ Error: {response.status_code} - {response.text}"
