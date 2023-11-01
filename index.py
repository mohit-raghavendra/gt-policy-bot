import pinecone
import tqdm
import yaml

import numpy as np
import pandas as pd

from typing import List
from pinecone.index import Index

def connect_index(index_name: str, api_key: str, env:str, embedding_dimension: int, delete_existing: bool = False):
    pinecone.init(
        api_key=api_key,
        environment=env
    )

    if index_name in pinecone.list_indexes() and delete_existing:
        pinecone.delete_index(index_name)

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=embedding_dimension)

    index = pinecone.Index(index_name)
    return index


def upsert_docs(index: Index, df: pd.DataFrame, embeddings_col: str, metadata_cols: List[str], batch_size: int=2):
    embeddings = df[embeddings_col].apply(lambda x: eval(x)).to_list()
    metadata = df[metadata_cols].to_dict(orient='records')
    num_docs = len(embeddings)
    for i in tqdm.tqdm(range(0, num_docs, batch_size)):
        embeddings_batch = embeddings[i: i+batch_size]
        ids_batch = df.index[i: i+batch_size].astype(str).to_list()
        metadata_batch = metadata[i: i+batch_size]
        
        to_upsert = list(zip(ids_batch, embeddings_batch, metadata_batch))
        print(to_upsert[0])
        index.upsert(vectors=to_upsert)


if __name__ == '__main__':
    data_path = 'data/music/'
    file_name = 'english_recent_pop_rnb_songs_embedding'
    format = '.csv'
    
    config_path = 'config.yml'
    
    embedding_dimension = 768
    index_name = 'song-search'
    delete_existing = True

    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    print(config)
    df = pd.read_csv(data_path + file_name + format, index_col = 0)
    print(df)

    index = connect_index(index_name,
                          config['pinecone']['api-key'], 
                          config['pinecone']['environment'], 
                          embedding_dimension, 
                          delete_existing
                          )
    
    print(pinecone.describe_index(index_name))

    upsert_docs(index, df, 'embeddings', ['artist', 'title'])
