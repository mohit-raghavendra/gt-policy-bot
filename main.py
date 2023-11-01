import yaml

from vectorise import Vectorizer
from index import VectorDB

if __name__ == '__main__':

    config_path = 'config.yml'
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    print(config)

    data_path = config['paths']['data_path']
    project = config['paths']['project']

    vectorizer = Vectorizer(config['sentence-transformers']['model-name'])  
    query = "Latest revision data of the student code of conduct"
    query_embedding = vectorizer.get_query_embedding(query).tolist()
    print(len(query_embedding), type(query_embedding))
    
    vector_db = VectorDB(config['pinecone']['index-name'])
    vector_db.connect_index(config['pinecone']['api-key'], 
                            config['pinecone']['environment'], 
                            config['sentence-transformers']['embedding-dimension'], 
                            False)
    
    results = vector_db.query([query_embedding], top_k=5)
    for doc in results:
        print("*****************************")
        print(doc)
