import os
from typing import List

import pinecone
from encoding import build_sparse_vecs
from tqdm.auto import tqdm


def builder(tokenizer, sparse_model, dense_model, records: List):
    """
    Transform a list of records from data into the format required to insert into pinecone index
    """
    upserts = []
    ids = [x["id"] for x in records]
    contexts = [x["context"] for x in records]
    dense_vecs = dense_model.encode(contexts).tolist()
    sparse_vecs = build_sparse_vecs(tokenizer, contexts, sparse_model)
    for _id, dense_vec, sparse_vec, context in zip(
        ids, dense_vecs, sparse_vecs, contexts
    ):
        # extract columns where there are non-zero weights
        indices = sparse_vec.nonzero().squeeze().cpu().tolist()  # positions
        values = sparse_vec[indices].cpu().tolist()  # weights/scores
        sparse_values = {"indices": indices, "values": values}
        metadata = {"context": context}
        upserts.append(
            {
                "id": _id,
                "values": dense_vec,
                "sparse_values": sparse_values,
                "metadata": metadata,
            }
        )
    return upserts


def upsert_data(data, index, tokenizer, sparse_model, dense_model, batch_size=64):
    for i in tqdm(range(0, len(data), batch_size)):
        # extract batch of data
        index.upsert(
            builder(
                tokenizer=tokenizer,
                sparse_model=sparse_model,
                dense_model=dense_model,
                records=data[i : i + batch_size],
            )
        )
    print(len(data), index.describe_index_stats())


def init_pinecone_connection():
    api_key = os.getenv("PINECONE_API_KEY") or "YOUR_API_KEY"
    env = os.getenv("PINECONE_ENVIRONMENT") or "YOUR_ENVIRONMENT"
    pinecone.init(api_key=api_key, environment=env)


def get_index(index_name, dim, create_index: bool = False):
    init_pinecone_connection()
    if create_index:
        pinecone.create_index(
            index_name, dimension=dim, metric="dotproduct", pod_type="s1"
        )
        index = pinecone.GRPCIndex(index_name)
    else:
        index = pinecone.GRPCIndex(index_name)
    return index
