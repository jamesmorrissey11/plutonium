import itertools
from collections import Counter
from typing import List, Tuple

import pandas as pd
import pinecone
import tensorflow.keras.backend as K
from tqdm import tqdm
from utility import add_to_pred_list, add_to_true_list


def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


def generate_upload_items(embeddings, df: pd.DataFrame) -> List[Tuple]:
    print("Generating items to upload...")
    items_to_upload = []
    for i, embedding in tqdm(zip(df.iterrows(), embeddings), total=len(embeddings)):
        benign_or_attack = i[1]["Label"][:3]
        items_to_upload.append((benign_or_attack + "_" + str(i[0]), embedding.tolist()))
    return items_to_upload


def upsert_batches(items_to_upload, index: pinecone.Index):
    print("Upserting batches...")
    NUMBER_OF_ITEMS = len(items_to_upload)
    for batch in tqdm(chunks(items_to_upload[:NUMBER_OF_ITEMS], 50)):
        index.upsert(vectors=batch)


def batch_query_results(test_vector, index: pinecone.Index):
    query_results = []
    for xq in test_vector.tolist():
        query_res = index.query(xq, top_k=50)
        query_results.append(query_res)
    return query_results
