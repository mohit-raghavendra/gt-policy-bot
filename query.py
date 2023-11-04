import yaml

import gradio as gr

from vectorise import Vectorizer
from index import VectorDB

TOP_K = 5

config_path = 'config.yml'
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

print(config)

data_path = config['paths']['data_path']
project = config['paths']['project']

vectorizer = Vectorizer(config['sentence-transformers']['model-name'])  
# query = "Latest revision data of the student code of conduct"

vector_db = VectorDB(config['pinecone']['index-name'])
vector_db.connect_index(config['pinecone']['api-key'], 
                        config['pinecone']['environment'], 
                        config['sentence-transformers']['embedding-dimension'], 
                        False)


def run_query(query: str):
    print(query)
    query_embedding = vectorizer.get_query_embedding(query).tolist()
    closest_documents= vector_db.query([query_embedding], top_k=TOP_K)
    return closest_documents

    # demo = gr.Interface(
    #     fn=run_query, 
    #     inputs="text", 
    #     outputs=["text"]*TOP_K,
    #     title="Code of Conduct Search",
    #     description="Search for code of conduct snippets using semantic search."
    # )

    # demo.launch() 


# if __name__ == '__main__':

      