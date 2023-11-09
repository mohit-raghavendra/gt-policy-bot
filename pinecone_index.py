import os
import pinecone

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

        # Check for existence of keys
        assert os.getenv('PINECONE_KEY') != None
        assert os.getenv('PINECONE_ENV') != None

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
    
    def get_embedding_model(self):
        return self._embeddingModel
    
    def get_index_name(self):
        return self.index_name

    def query(self, query: str, top_k: int = 5) -> List[str]:
        docsearch = Pinecone.from_existing_index(self.index_name,
                                                 self._embeddingModel)
        res = docsearch.similarity_search(query, k=top_k)

        return [doc.page_content for doc in res]
