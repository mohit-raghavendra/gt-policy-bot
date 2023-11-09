import time
import yaml
from langchain.vectorstores.pinecone import Pinecone
from pinecone_index import PinceconeIndex
import pandas as pd

class VectorStore:
    def __init__(self, configFilePath="config.yml"):
        self._config_path = configFilePath

    def get_vector_store(self):
        with open(self._config_path, 'r') as file:
            config = yaml.safe_load(file)

        print("Config file loaded ...")
        print(config)

        data_path = config['paths']['data_path']
        project = config['paths']['project']
        format = '.csv' # this can be added to config ?

        index_name = config['pinecone']['index-name']
        embedding_model = config['sentence-transformers'][
            'model-name']
        embedding_dimension = config['sentence-transformers'][
            'embedding-dimension']
        delete_existing = True

        file_path_embedding = data_path + project + format
        df = pd.read_csv(file_path_embedding, index_col=0)
        print("Loaded embeddings ...")
        print(df.head())

        start_time = time.time()
        index = PinceconeIndex(index_name, embedding_model)
        index.connect_index(embedding_dimension, delete_existing)
        index.upsert_docs(df, 'chunks')
        end_time = time.time()
        print(f'Indexing took {end_time - start_time} seconds')

        vectorstore = Pinecone.from_existing_index(index_name, index.get_embedding_model())

        return vectorstore
