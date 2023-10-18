import argparse
import os
import sys

import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Model

sys.path.append("/Users/jamesmorrissey/Github/plutonium/src")

from core.cloud.pc import get_pinecone_index, init_pinecone_connection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--model_dir", type=str, default="/Users/jamesmorrissey/Github/plutonium/models"
    )
    parser.add_argument("--cleaned_data", type=str, default="result23.csv")
    parser.add_argument("--index_name", type=str, default="it-threats")
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--create_index", action="store_true")
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

    print("Loading data...")
    data = pd.read_csv(
        os.path.join(args.data_dir, "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv")
    )
    data_23_cleaned = pd.read_csv(os.path.join(args.data_dir, args.cleaned_data))

    layer_name = "dense"
    model = keras.models.load_model(
        os.path.join(args.model_dir, "it_threat_model.model")
    )
    intermediate_layer_model = Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output
    )
    print(model.summary())
