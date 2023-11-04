import yaml

import gradio as gr

from vectorise import Vectorizer
from index import VectorDB

TOP_K = 5

if __name__ == '__main__':

    config_path = 'config.yml'
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    print(config)

    data_path = config['paths']['data_path']
    project = config['paths']['project']

    vectorizer = Vectorizer(config['sentence-transformers']['model-name'])  
    
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

    demo = gr.Interface(
        fn=run_query, 
        inputs=gr.Textbox(placeholder="Enter your question..."), 
        outputs=[gr.Textbox(label=f'Document {i+1}') for i in range(TOP_K)],
        title="GT Student Code of Conduct Bot",
        description="Get LLM-powered answers to questions about the Georgia Tech Student Code of Conduct."
    )

    demo.launch()   