import argparse
import os
import sys
from collections import Counter
from typing import List, Tuple

import pandas as pd
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.models import Model
from tqdm import tqdm

sys.path.append("/Users/jamesmorrissey/Github/plutonium/src")

from index import get_predictions, upsert_batches
from metrics import per_class_accuracy, plot_confusion_matrix, print_metrics

from core.cloud.pc import get_pinecone_index, init_pinecone_connection


def generate_items(embeddings, df: pd.DataFrame) -> List[Tuple]:
    items_to_upload = []
    for i, embedding in tqdm(zip(df.iterrows(), embeddings), total=len(embeddings)):
        benign_or_attack = i[1]["Label"][:3]
        items_to_upload.append((benign_or_attack + "_" + str(i[0]), embedding.tolist()))
    return items_to_upload


def build_embedding_model(args):
    layer_name = "dense"
    model = keras.models.load_model(
        os.path.join(args.model_dir, "it_threat_model.model")
    )
    intermediate_layer_model = Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output
    )
    return intermediate_layer_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--model_dir", type=str, default="/Users/jamesmorrissey/Github/plutonium/models"
    )
    parser.add_argument("--index_name", type=str, default="it-threats")
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--create_index", action="store_true")
    parser.add_argument("--upsert_data", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    init_pinecone_connection()
    index = get_pinecone_index(
        index_name=args.index_name,
        dimension=args.dimension,
        metric=args.metric,
        create_index=args.create_index,
    )

    data_23_cleaned = pd.read_csv(os.path.join(args.data_dir, "result23.csv"))
    data_22_cleaned = pd.read_csv(os.path.join(args.data_dir, "result22022018.csv"))
    data_sample = data_22_cleaned[-2000:]

    embedding_model = build_embedding_model(args)
    embeddings = embedding_model.predict(K.constant(data_23_cleaned.iloc[:, :-1]))
    items_to_upload = generate_items(embeddings, df=data_23_cleaned)

    if args.upsert_data:
        upsert_batches(items_to_upload, index)
        items_to_upload.clear()

    print(index.describe_index_stats())

    y_true, y_pred = get_predictions(
        data_sample=data_sample, embedding_model=embedding_model
    )
    plot_confusion_matrix(y_true=y_true, y_pred=y_pred)
    print_metrics(y_true=y_true, y_pred=y_pred)
    per_class_acc = per_class_accuracy(y_true=y_true, y_pred=y_pred)
    print(f"Per Class Accuracy: \n {per_class_acc}")
