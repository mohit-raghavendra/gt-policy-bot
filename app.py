import os
import yaml

import gradio as gr

from vectorise import Vectorizer
from index import VectorDB
from pinecone_index import PinceconeIndex

TOP_K = 5

if __name__ == '__main__':

    config_path = 'config.yml'
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    print(config)

    data_path = config['paths']['data_path']
    project = config['paths']['project']
    
    index_name = config['pinecone']['index-name']
    embedding_model = config['sentence-transformers']['model-name']
    embedding_dimension = config['sentence-transformers']['embedding-dimension']    

    index = PinceconeIndex(index_name, embedding_model)
    index.connect_index(embedding_dimension, False)

    def run_query(query: str):
        res = index.query(query, top_k=5)
        return res

    demo = gr.Interface(
        fn=run_query, 
        inputs=gr.Textbox(placeholder="Enter your question..."), 
        outputs=[gr.Textbox(label=f'Document {i+1}') for i in range(TOP_K)],
        title="GT Student Code of Conduct Bot",
        description="Get LLM-powered answers to questions about the Georgia Tech Student Code of Conduct."
    )

    demo.launch()   