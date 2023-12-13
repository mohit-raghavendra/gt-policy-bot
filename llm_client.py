import os

import google.generativeai as genai


class GeminiClient:
    def __init__(self):
        self.connect_client()

    def connect_client(self):
        if not os.getenv("GOOGLE_PALM_KEY"):
            raise Exception("Please set your Google AI Studio key")

        api_key = os.getenv("GOOGLE_PALM_KEY")
        genai.configure(api_key=api_key)

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
        ]

        defaults = {
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.95,
            "max_output_tokens": 1024,
        }

        self.model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=defaults,
            safety_settings=safety_settings,
        )

    def generate_text(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text
