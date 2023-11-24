import os

import google.generativeai as palm


class PalmClient:
    def __init__(self):
        self.connect_client()

    def connect_client(self):
        if (not os.getenv('GOOGLE_PALM_KEY')):
            raise Exception('Please set your Google MakerSuite API key')

        api_key = os.getenv('GOOGLE_PALM_KEY')
        palm.configure(api_key=api_key)

        safety_overrides = [
            {"category": "HARM_CATEGORY_DEROGATORY", "threshold": 4},
            {"category": "HARM_CATEGORY_TOXICITY", "threshold": 4},
            {"category": "HARM_CATEGORY_VIOLENCE", "threshold": 4},
            {"category": "HARM_CATEGORY_SEXUAL", "threshold": 4},
            {"category": "HARM_CATEGORY_MEDICAL", "threshold": 4},
            {"category": "HARM_CATEGORY_DANGEROUS", "threshold": 4}
            ]

        defaults = {
            'model': 'models/text-bison-001',
            'temperature': 0.7,
            'candidate_count': 1,
            'top_k': 40,
            'top_p': 0.95,
            'max_output_tokens': 1024,
            'stop_sequences': [],
            'safety_settings': safety_overrides,
        }

        self.defaults = defaults

    def generate_text(self, prompt: str) -> str:
        response = palm.generate_text(**self.defaults, prompt=prompt)
        return response.candidates[0]['output']
