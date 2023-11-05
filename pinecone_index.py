import os
import pinecone
import time
import yaml

import pandas as pd

from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.pinecone import Pinecone

from typing import List


class PinceconeIndex:
    def __init__(self, index_name: str, model_name: str):
        self.index_name = index_name
        self._embeddingModel = HuggingFaceEmbeddings(model_name=model_name)

    def connect_index(self, embedding_dimension: int,
                      delete_existing: bool = False):
        index_name = self.index_name

        pinecone.init(
            api_key=os.getenv('PINECONE_KEY'),
            environment=os.getenv('PINECONE_ENV'),
        )

        if index_name in pinecone.list_indexes() and delete_existing:
            pinecone.delete_index(index_name)

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=embedding_dimension)

        index = pinecone.Index(index_name)

        pinecone.describe_index(index_name)
        self._index = index

    def upsert_docs(self, df: pd.DataFrame, text_col: str):
        loader = DataFrameLoader(df, page_content_column=text_col)
        docs = loader.load()
        Pinecone.from_documents(docs, self._embeddingModel,
                                index_name=self.index_name)

    def query(self, query: str, top_k: int = 5) -> List[str]:
        docsearch = Pinecone.from_existing_index(self.index_name,
                                                 self._embeddingModel)
        res = docsearch.similarity_search(query, k=top_k)

        return [doc.page_content for doc in res]


if __name__ == '__main__':
    config_path = 'config.yml'
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    print(config)

    data_path = config['paths']['data_path']
    project = config['paths']['project']
    format = '.csv'

    index_name = config['pinecone']['index-name']
    embedding_model = config['sentence-transformers'][
        'model-name']
    embedding_dimension = config['sentence-transformers'][
        'embedding-dimension']
    delete_existing = True

    file_path_embedding = data_path+project+format
    df = pd.read_csv(file_path_embedding, index_col=0)
    print(df.head())

    start_time = time.time()
    index = PinceconeIndex(index_name, embedding_model)
    index.connect_index(embedding_dimension, delete_existing)
    index.upsert_docs(df, 'chunks')
    end_time = time.time()
    print(f'Indexing took {end_time - start_time} seconds')

    index = PinceconeIndex(index_name, embedding_model)
    index.connect_index(embedding_dimension, delete_existing=False)

    query = "When was the student code of conduct last revised?"
    res = index.query(query, top_k=5)

    assert len(res) == 5
    print(res)
