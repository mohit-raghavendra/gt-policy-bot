import re

import pandas as pd

from pdfminer.high_level import extract_text


class DocumentParser:
    def __init__(self, filename, chunk_size=200):
        self.filename = filename
        self.text = ""
        self.chunk_size = chunk_size
        self.chunks = []
        self.num_chunks = 0

    def _read_pdf(self):
        self.text = extract_text(self.filename)
        # print(self.text)

    def _clean_text(self):
        self.text = " ".join(self.text.split())
        self.text = re.sub(r"[^a-zA-Z0-9.!?-]", " ", self.text)

    def _chunk_text(self):
        chunk_curr = ""

        for letter in self.text:
            chunk_curr += letter
            if letter == "." and len(chunk_curr) >= self.chunk_size:
                self.chunks.append(chunk_curr)
                chunk_curr = ""
        if chunk_curr:
            self.chunks.append(chunk_curr)

        self.num_chunks = len(self.chunks)

    def _save_as_csv(self, csv_filepath: str, col_name):
        df = pd.DataFrame(self.chunks, columns=[col_name])
        df.to_csv(csv_filepath)

    def run(self, csv_filepath: str, col_name: str):
        self._read_pdf()
        self._clean_text()
        self._chunk_text()
        self._save_as_csv(csv_filepath, col_name)


if __name__ == "__main__":
    dp = DocumentParser("./data/code_of_conduct/code_of_conduct.pdf", 500)
    csv_filepath = "./data/code_of_conduct/code_of_conduct.csv"

    dp.run(csv_filepath=csv_filepath, col_name="chunks")
    df = pd.read_csv(csv_filepath, index_col=0)
    print(df)
