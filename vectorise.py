import tqdm

import pandas as pd

from typing import List

from sentence_transformers import SentenceTransformer

BATCH_SIZE = 2

def load_model(model_name: str):
    model = SentenceTransformer(model_name)
    return model

def embed_docs(df: pd.DataFrame, data_col: str, model):
    docs = df[data_col]
    num_docs = len(docs)
    print(model)
    embeddings = []
    for i in tqdm.tqdm(range(0, num_docs, BATCH_SIZE)):
        docs_batch = docs[i: i+BATCH_SIZE].to_list()
        vectors_batch = model.encode(docs_batch).tolist()
        embeddings.append(vectors_batch)

    return [embedding for batch in embeddings for embedding in batch]


if __name__ == '__main__':
    subset = 10
    data_path = 'data/music/'
    file_name = 'english_recent_pop_rnb_songs'
    format = '.csv'

    df = pd.read_csv(data_path + file_name + format).head(subset)
    df.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
    
    model = load_model('thenlper/gte-base')
    embeddings = embed_docs(df, 'lyrics', model)
    print(len(embeddings))
    df['embeddings'] = embeddings
    print(df)

    file_path_embedding = data_path+file_name+'_embedding'+format
    df.to_csv(file_path_embedding)

    df_read = pd.read_csv(file_path_embedding, index_col=0)
    print(df_read)