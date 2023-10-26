import argparse
import itertools
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pinecone
import seaborn as sns
import tensorflow.keras.backend as K
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from tensorflow import keras
from tensorflow.keras.models import Model
from tqdm import tqdm


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


def load_encoding_model(model_name: str):
    model = keras.models.load_model(model_name)
    model.summary()

    # Select the first layer
    layer_name = "dense"
    intermediate_layer_model = Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output
    )
    return intermediate_layer_model


def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


def upload_items(data_23_cleaned, encoded_features):
    items_to_upload = []
    for i, res in tqdm(
        zip(data_23_cleaned.iterrows(), encoded_features), total=len(encoded_features)
    ):
        benign_or_attack = i[1]["Label"][:3]
        items_to_upload.append((benign_or_attack + "_" + str(i[0]), res.tolist()))
    for batch in chunks(items_to_upload[: len(items_to_upload)], 50):
        index.upsert(vectors=batch)
    items_to_upload.clear()
    index.describe_index_stats()


def load_cleaned_data(data_dir):
    data_23_cleaned = pd.read_csv(os.path.join(data_dir, "result23.csv"))
    return data_23_cleaned


if __name__ == "__main__":
    args = argparse.Namespace(
        index_name="it-threats",
        data_dir="Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
        num_upload_items=None,
    )
    init_pinecone_connection()
    index = pinecone.Index("it-threats")
    data = load_cleaned_data(args.data_dir)
    encoding_model = load_encoding_model("it_threat_model.model")
    encoded_features = encoding_model.predict(K.constant(data.iloc[:, :-1]))
