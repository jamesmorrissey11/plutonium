from collections import Counter

import pandas as pd
from tqdm import tqdm


def create_upload_items(model_res, df: pd.DataFrame):
    items_to_upload = []
    for i, res in tqdm(zip(df.iterrows(), model_res), total=len(model_res)):
        benign_or_attack = i[1]["Label"][:3]
        items_to_upload.append((benign_or_attack + "_" + str(i[0]), res.tolist()))
    return items_to_upload


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
