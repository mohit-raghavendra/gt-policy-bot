import re
from pdfminer.high_level import extract_text

class DocumentParser:
    def __init__(self, filename, chunk_size=200):
        self.filename = filename
        self.text = ""
        self.chunk_size = chunk_size
        self.chunks = []

    def read_pdf(self):
        self.text = extract_text(self.filename)
    
    def clean_text(self):
        self.text = ' '.join(self.text.split())
        self.text = re.sub(r'[^a-zA-Z0-9.!?-]', ' ', self.text)

    def chunk_text(self):
        chunk_curr = ""

        for letter in self.text:
            chunk_curr += letter
            if letter == "." and len(chunk_curr) >= self.chunk_size:
                self.chunks.append(chunk_curr)
                chunk_curr = ""
        if chunk_curr:
            self.chunks.append(chunk_curr)

if __name__ == "__main__":
    dp = DocumentParser("./test.pdf", 500)
    dp.read_pdf()
    dp.clean_text()
    dp.chunk_text()

    # print(dp.chunks[10])
    