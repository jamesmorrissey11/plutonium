import argparse
from typing import Dict

import torch
from datasets import load_dataset
from index import get_index, upsert_data
from pinecone.core.client.model.query_response import QueryResponse
from queries import query_index
from sentence_transformers import SentenceTransformer
from splade.models.transformer_rep import Splade
from transformers import AutoTokenizer
from utility import generate_data


def init_sparse_model(id) -> Splade:
    """
    Initialize a learned sparse embedding model called SPLADE
    - The model takes tokenized inputs that are built using a tokenizer initialized with the same model ID.
    """
    sparse_model = Splade(id, agg="max")
    sparse_model.to("cpu")  # move to GPU if possible
    sparse_model.eval()
    return sparse_model


def init_models(sparse_model, dense_model, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(sparse_model)
    dense_model = SentenceTransformer(dense_model, device=device)
    sparse_model = init_sparse_model(id=sparse_model)
    return tokenizer, sparse_model, dense_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_name", type=str, default="pubmed-splade")
    parser.add_argument(
        "--sparse_model", type=str, default="naver/splade-cocondenser-ensembledistil"
    )
    parser.add_argument("--dense_model", type=str, default="msmarco-bert-base-dot-v5")
    parser.add_argument("--char_limit", type=int, default=384)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--upsert_data", action="store_true")


def get_sparse_emb_dim(sparse_model, tokenizer, sample):
    tokens = tokenizer(sample, return_tensors="pt")
    with torch.no_grad():
        sparse_emb = sparse_model(d_kwargs=tokens.to("cpu"))["d_rep"].squeeze()
    sparse_emb_dim = sparse_emb.shape
    return sparse_emb_dim


if __name__ == "__main__":
    args = parse_args()
    pubmed = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    data = generate_data(raw_data=pubmed, char_limit=args.char_limit)
    tokenizer, sparse_model, dense_model = init_models(
        sparse_model=args.sparse_model, dense_model=args.dense_model
    )

    sparse_vector_dim = get_sparse_emb_dim(
        sparse_model=sparse_model, tokenizer=tokenizer, sample=data[0]["context"]
    )
    dense_vector_dim = dense_model.get_sentence_embedding_dimension()

    index = get_index(index_name=args.index_name, dim=dense_vector_dim)
    if args.upsert_data:
        upsert_data(
            data=data,
            index=index,
            tokenizer=tokenizer,
            sparse_model=sparse_model,
            dense_model=dense_model,
        )

    xcs: Dict[str, QueryResponse] = {
        "balanced": query_index(
            query=args.query,
            method="balanced",
            tokenizer=tokenizer,
            sparse_model=sparse_model,
            dense_model=dense_model,
            index=index,
        ),
        "dense": query_index(
            query=args.query,
            method="dense",
            tokenizer=tokenizer,
            sparse_model=sparse_model,
            dense_model=dense_model,
            index=index,
        ),
        "sparse": query_index(
            query=args.query,
            method="sparse",
            tokenizer=tokenizer,
            sparse_model=sparse_model,
            dense_model=dense_model,
            index=index,
        ),
    }
