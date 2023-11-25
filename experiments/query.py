import yaml

from pinecone_index import PinceconeIndex


TOP_K = 5

config_path = "config.yml"
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

print(config)

data_path = config["paths"]["data_path"]
project = config["paths"]["project"]

index_name = config["pinecone"]["index-name"]
embedding_model = config["sentence-transformers"]["model-name"]
embedding_dimension = config["sentence-transformers"]["embedding-dimension"]

index = PinceconeIndex(index_name, embedding_model)
index.connect_index(embedding_dimension, False)


def run_query(query: str, top_k=TOP_K):
    res = index.query(query, top_k=top_k)
    return res
