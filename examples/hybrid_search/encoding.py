import torch


def encode(text: str, tokenizer, sparse_model, dense_model):
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


def build_sparse_vecs(tokenizer, contexts, sparse_model):
    input_ids = tokenizer(contexts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        sparse_vecs = sparse_model(d_kwargs=input_ids.to("cpu"))["d_rep"].squeeze()
    return sparse_vecs
