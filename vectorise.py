import tqdm
import yaml

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

BATCH_SIZE = 2

class Vectorizer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.batch_size = BATCH_SIZE

    def get_query_embedding(self, query: str) -> np.ndarray:
        return self.model.encode(query)
                              
    def get_embeddings(self, df: pd.DataFrame, data_col: str):
        docs = df[data_col]
        num_docs = len(docs)
        embeddings = []
        for i in tqdm.tqdm(range(0, num_docs, self.batch_size)):
            docs_batch = docs[i: i + self.batch_size].to_list()
            vectors_batch = self.model.encode(docs_batch).tolist()
            embeddings.append(vectors_batch)

        embeddings_flattened = [embedding for batch in embeddings for embedding in batch]

        assert len(embeddings_flattened) == num_docs
        return embeddings_flattened

    def embed_docs(self, df: pd.DataFrame, data_col: str) -> pd.DataFrame:
        embeddings = self.get_embeddings(df, data_col)
        df['embeddings'] = embeddings
        
        return df

    def run(self, configFilePath="config.yml"):
        with open(configFilePath, 'r') as file:
            config = yaml.safe_load(file)
        print("Config File Loaded ...")
        print(config)

        data_path = config['paths']['data_path']
        project = config['paths']['project']
        format = '.csv'

        data_col_name = 'chunks'
        df = pd.read_csv(data_path + project + format)
        df.drop(labels=['Unnamed: 0'], axis=1, inplace=True)

        vectorizer = Vectorizer(config['sentence-transformers']['model-name'])
        df_embeddings = vectorizer.embed_docs(df, data_col_name)
        print("Creation of embedding completed ...")
        print(df_embeddings.head())

        file_path_embedding = data_path + project + '_embedding' + format
        df_embeddings.to_csv(file_path_embedding)

        df_read = pd.read_csv(file_path_embedding, index_col=0)
        assert len(df_read) == len(df_embeddings)
        print(file_path_embedding + "created ...")
    