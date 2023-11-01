import pinecone
import tqdm
import yaml

import numpy as np
import pandas as pd

from typing import List
from pinecone.index import Index

class VectorDB:
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.index = None

    def connect_index(self, api_key: str, env:str, embedding_dimension: int, delete_existing: bool = False):
        index_name = self.index_name

        pinecone.init(
            api_key=api_key,
            environment=env
        )

        if index_name in pinecone.list_indexes() and delete_existing:
            pinecone.delete_index(index_name)

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=embedding_dimension)

        index = pinecone.Index(index_name)

        pinecone.describe_index(index_name)
        self.index = index

    def upsert_docs(self, df: pd.DataFrame, embeddings_col: str, metadata_cols: List[str], batch_size: int=2):
        embeddings = df[embeddings_col].apply(lambda x: eval(x)).to_list()
        metadata = df[metadata_cols].to_dict(orient='records')
        num_docs = len(embeddings)
        print(f'Number of documents: {num_docs}')
        for i in tqdm.tqdm(range(0, num_docs, batch_size)):
            embeddings_batch = embeddings[i: i+batch_size]
            ids_batch = df.index[i: i+batch_size].astype(str).to_list()
            metadata_batch = metadata[i: i+batch_size]
            
            to_upsert = list(zip(ids_batch, embeddings_batch, metadata_batch))
            self.index.upsert(vectors=to_upsert)

    def query(self, query_embedding: np.ndarray, top_k: int=5):
        results = self.index.query(query_embedding, top_k=top_k, include_metadata=True)
        data = [match['metadata']['chunks'] for match in results['matches']]
        return data
    
if __name__ == '__main__':
    config_path = 'config.yml'
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    print(config)

    data_path = config['paths']['data_path']
    project = config['paths']['project']
    format = '.csv'

    
    index_name = config['pinecone']['index-name']
    embedding_dimension = config['sentence-transformers']['embedding-dimension']    
    delete_existing = True

    file_path_embedding = data_path+project+'_embedding'+format
    df = pd.read_csv(file_path_embedding, index_col = 0)
    print(df.head())

    vector_db = VectorDB(index_name)
    vector_db.connect_index(config['pinecone']['api-key'], 
                            config['pinecone']['environment'], 
                            embedding_dimension, 
                            delete_existing)


    vector_db.upsert_docs(df, 'embeddings', ['chunks'])
