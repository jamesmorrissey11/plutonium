import itertools
from collections import Counter

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


def upsert_batches(items_to_upload, index: pinecone.Index):
    NUMBER_OF_ITEMS = len(items_to_upload)
    for batch in chunks(items_to_upload[:NUMBER_OF_ITEMS], 50):
        index.upsert(vectors=batch)


def batch_query_results(test_vector, index: pinecone.Index):
    query_results = []
    for xq in test_vector.tolist():
        query_res = index.query(xq, top_k=50)
        query_results.append(query_res)
    return query_results


def get_predictions(data_sample, embedding_model, index, batch_size=100):
    y_true = []
    y_pred = []

    for i in tqdm(range(0, len(data_sample), batch_size)):
        test_data = data_sample.iloc[i : i + batch_size, :]
        # Create vector embedding using the model
        test_vector = embedding_model.predict(K.constant(test_data.iloc[:, :-1]))

        # Query using the vector embedding
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
