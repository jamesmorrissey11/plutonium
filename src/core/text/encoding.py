from typing import List

import torch
from sentence_transformers import SentenceTransformer
from splade.models.transformer_rep import Splade
from transformers import AutoTokenizer


def init_sparse_model(id) -> Splade:
    """
    Initialize a learned sparse embedding model called SPLADE
    - The model takes tokenized inputs that are built using a tokenizer initialized with the same model ID.
    """
    sparse_model = Splade(id, agg="max")
    sparse_model.to("cpu")  # move to GPU if possible
    sparse_model.eval()
    return sparse_model


def get_sparse_emb_dim(sparse_model, tokenizer, sample):
    tokens = tokenizer(sample, return_tensors="pt")
    with torch.no_grad():
        sparse_emb = sparse_model(d_kwargs=tokens.to("cpu"))["d_rep"].squeeze()
    sparse_emb_dim = sparse_emb.shape
    return sparse_emb_dim


def dense_and_sparse_encoding(
    text: str,
    tokenizer: AutoTokenizer,
    sparse_model: Splade,
    dense_model: SentenceTransformer,
):
    """
    Construct dense and sparse vectors for a given text
    """
    sparse_model_inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        sparse_vec = sparse_model(d_kwargs=sparse_model_inputs.to("cpu"))[
            "d_rep"
        ].squeeze()
    indices = sparse_vec.nonzero().squeeze().cpu().tolist()
    values = sparse_vec[indices].cpu().tolist()
    sparse_dict = {"indices": indices, "values": values}
    dense_vec = dense_model.encode(text).tolist()
    return dense_vec, sparse_dict


def build_sparse_vecs(tokenizer: AutoTokenizer, contexts, sparse_model: Splade):
    input_ids = tokenizer(contexts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        sparse_vecs = sparse_model(d_kwargs=input_ids.to("cpu"))["d_rep"].squeeze()
    return sparse_vecs


def hybrid_encoding_builder(tokenizer, sparse_model, dense_model, records: List):
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
