import pinecone

import yaml

import pandas as pd

from utils.parsedocument import DocumentParser
from vectorise import embed_docs, load_model
from index import connect_index, upsert_docs

if __name__ == '__main__':

    project = 'code_of_conduct'
    data_path = './data/'+project+'/'
    code_of_conduct_filename = project+'.pdf'
    dp = DocumentParser(data_path+code_of_conduct_filename, 500)
    dp.read_pdf()
    dp.clean_text()
    dp.chunk_text()

    csv_filepath = data_path+project+".csv"
    print(dp.chunks[10])

    data_col_name = 'chunks'

    dp.save_as_csv(csv_filepath, data_col_name)


    df = pd.read_csv(data_path + project + '.csv')
    df.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
    
    model = load_model('thenlper/gte-base')
    embeddings = embed_docs(df, data_col_name, model)
    print(len(embeddings))
    df['embeddings'] = embeddings
    print(df)

    file_path_embedding = data_path+project+'_embedding'+'.csv'
    df.to_csv(file_path_embedding)

    df_read = pd.read_csv(file_path_embedding, index_col=0)
    print(df_read.head())

    config_path = 'config.yml'
    
    embedding_dimension = 768
    index_name = 'song-search'
    delete_existing = True

    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    print(config)
    df = pd.read_csv(data_path + project + '_embedding' + '.csv', index_col = 0)

    index = connect_index(index_name,
                          config['pinecone']['api-key'], 
                          config['pinecone']['environment'], 
                          embedding_dimension, 
                          delete_existing
                          )
    
    print(pinecone.describe_index(index_name))

    upsert_docs(index, df, 'embeddings', ['chunks'])