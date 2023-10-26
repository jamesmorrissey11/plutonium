import itertools
import os

import pandas as pd
import pinecone
from connector import PineconeConnector
from model import KerasEncoder
from tqdm import tqdm


class Generator:
    def __init__(
        self, args, encoder_model: KerasEncoder, pinecone_connector: PineconeConnector
    ):
        self.args = args
        self.encoder_model = encoder_model
        self.pinecone_connector = pinecone_connector

        self.data = self.load_data(args.data_dir, args.file_name)
        self.upload_data = self.build_upload_data()

    def load_data(self, data_dir: str, file_name: str) -> pd.DataFrame:
        data = pd.read_csv(
            os.path.join(data_dir, file_name),
        )
        return data

    def build_upload_data(self) -> list:
        upload_data = []
        encodings = self.encoder_model.encode_features(df=self.data)
        for i, res in tqdm(zip(self.data.iterrows(), encodings, total=len(encodings))):
            benign_or_attack = i[1]["Label"][:3]
            upload_data.append((benign_or_attack + "_" + str(i[0]), res.tolist()))
        return upload_data

    def upload_data_to_pinecone(self, n_items, index_name) -> None:
        index = pinecone.Index(index_name)
        for batch in self.chunks(self.upload_data[:n_items], 50):
            index.upsert(vectors=batch)
        self.upload_data.clear()
        index.describe_index_stats()

    @staticmethod
    def chunks(iterable, batch_size=100):
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))
