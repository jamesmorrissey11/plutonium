import argparse
import os
import sys
from collections import Counter
from typing import List, Tuple

import pandas as pd
import pinecone
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.models import Model
from tqdm import tqdm

sys.path.append("/Users/jamesmorrissey/Github/plutonium/src")

from index import generate_upload_items, upsert_batches
from metrics import per_class_accuracy, plot_confusion_matrix, print_metrics

from core.cloud.pc import get_pinecone_index, init_pinecone_connection


def build_embedding_model(args):
    layer_name = "dense"
    model = keras.models.load_model(
        os.path.join(args.model_dir, "it_threat_model.model")
    )
    intermediate_layer_model = Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output
    )
    return intermediate_layer_model


def get_predictions(data_sample, embedding_model, index, batch_size=100):
    y_true = []
    y_pred = []
    for i in tqdm(range(0, len(data_sample), batch_size)):
        test_data = data_sample.iloc[i : i + batch_size, :]
        test_vector = embedding_model.predict(K.constant(test_data.iloc[:, :-1]))
        query_results = []
        for xq in test_vector.tolist():
            query_res = index.query(xq, top_k=50)
            query_results.append(query_res)
        for label, res in zip(test_data.Label.values, query_results):
            y_true.append(add_to_true_list(label))
            counter = Counter(match.id.split("_")[0] for match in res.matches)
            y_pred.append(add_to_pred_list(counter))

    return y_true, y_pred


def add_to_true_list(label: str):
    if label == "Benign":
        return 0
    else:
        return 1


def add_to_pred_list(counter: Counter):
    if counter["Bru"] or counter["SQL"]:
        return 1
    else:
        return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--model_dir", type=str, default="/Users/jamesmorrissey/Github/plutonium/models"
    )
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--create_index", action="store_true")
    parser.add_argument("--upsert_data", action="store_true")
    args = parser.parse_args()
    return args


CREATE_INDEX = False
if __name__ == "__main__":
    args = parse_args()
    init_pinecone_connection()
    if CREATE_INDEX:
        pinecone.create_index(name="it-threats", dimension=128, metric="euclidean")
        index = pinecone.Index("it-threats")
    else:
        index = pinecone.Index("it-threats")

    data_23_cleaned = pd.read_csv(os.path.join(args.data_dir, "result23.csv"))
    data_22_cleaned = pd.read_csv(os.path.join(args.data_dir, "result22022018.csv"))
    data_sample = data_22_cleaned[-2000:]

    embedding_model = build_embedding_model(args)
    embeddings = embedding_model.predict(K.constant(data_23_cleaned.iloc[:, :-1]))
    items_to_upload = generate_upload_items(embeddings, df=data_23_cleaned)

    if args.upsert_data:
        upsert_batches(items_to_upload, index)
        items_to_upload.clear()

    y_true, y_pred = get_predictions(
        data_sample=data_sample, embedding_model=embedding_model, index=index
    )
    plot_confusion_matrix(y_true=y_true, y_pred=y_pred)
    print_metrics(y_true=y_true, y_pred=y_pred)
    per_class_acc = per_class_accuracy(y_true=y_true, y_pred=y_pred)
    print(f"Per Class Accuracy: \n {per_class_acc}")
