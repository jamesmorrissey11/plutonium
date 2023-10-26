import os

import pinecone


class PineconeConnector:
    def __init__(self, args, index_name=None, dimension=None, metric=None):
        self.args = args
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.init_pinecone_connection()

    def create_pinecone_index(self, index_name, dimension, metric):
        try:
            pinecone.create_index(name=index_name, dimension=dimension, metric=metric)
            return pinecone.Index(index_name)
        except Exception as e:
            print(f"Unable to create index due to {e}")

    def get_pinecone_index(self, exists=True):
        if exists:
            return pinecone.Index(self.index_name)
        else:
            return self.create_pinecone_index(
                index_name=self.index_name,
                dimension=self.args.dimension,
                metric=self.args.metric,
            )

    @staticmethod
    def init_pinecone_connection():
        api_key = os.getenv("PINECONE_API_KEY")
        env = os.getenv("PINECONE_ENVIRONMENT")
        pinecone.init(api_key=api_key, environment=env)
