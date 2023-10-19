from encoding import encode


def balanced_query(index, dense_vector, sparse_vector):
    xc = index.query(
        vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=2,
        include_metadata=True,
    )
    return xc


def dense_query(index, dense_vector, sparse_vector):
    hdense, hsparse = hybrid_scale(dense_vector, sparse_vector, alpha=1.0)
    xc = index.query(
        vector=hdense,
        sparse_vector=hsparse,
        top_k=2,
        include_metadata=True,
    )
    return xc


def sparse_query(index, dense_vector, sparse_vector):
    hdense, hsparse = hybrid_scale(dense_vector, sparse_vector, alpha=0.0)
    xc = index.query(
        vector=hdense, sparse_vector=hsparse, top_k=2, include_metadata=True
    )
    return xc


def query_index(query, method, tokenizer, sparse_model, dense_model, index):
    dense, sparse = encode(
        text=query,
        tokenizer=tokenizer,
        sparse_model=sparse_model,
        dense_model=dense_model,
    )
    if method == "balanced":
        return balanced_query(index, dense, sparse)
    elif method == "dense":
        return dense_query(index, dense, sparse)
    elif method == "sparse":
        return sparse_query(index, dense, sparse)


def hybrid_scale(dense, sparse, alpha: float):
    """
    Weight the dense and sparse vectors by alpha and (1 - alpha) respectively
    - alpha == 0 -> pure sparse search
    - alpha == 1 -> pure dense search
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hsparse = {
        "indices": sparse["indices"],
        "values": [v * (1 - alpha) for v in sparse["values"]],
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse
