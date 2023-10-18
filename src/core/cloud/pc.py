import os

import pinecone


def init_pinecone_connection():
    api_key = os.getenv("PINECONE_API_KEY") or "YOUR_API_KEY"
    env = os.getenv("PINECONE_ENVIRONMENT") or "YOUR_ENVIRONMENT"
    pinecone.init(api_key=api_key, environment=env)


def get_pinecone_index(index_name, dimension, metric, create_index):
    if create_index:
        if index_name in pinecone.list_indexes():
            pinecone.delete_index(index_name)
        pinecone.create_index(name=index_name, dimension=dimension, metric=metric)
    else:
        index = pinecone.Index(index_name)
    return index
