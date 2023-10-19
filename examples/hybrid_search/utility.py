from typing import List


def generate_data(raw_data, char_limit) -> List:
    data = []
    for record in raw_data:
        chunks = chunker(record["context"]["contexts"], char_limit=char_limit)
        for i, context in enumerate(chunks):
            data.append({"id": f"{record['pubid']}-{i}", "context": context})
    return data


def chunker(contexts: List[str], char_limit: int = 384):
    """
    Splits a list of text contexts into smaller passages while ensuring that each passage does not exceed a character limit.
    """
    chunks = []
    chunk = []
    # Split each context into sentences
    all_contexts = " ".join(contexts).split(".")
    for context in all_contexts:
        chunk.append(context)
        if len(chunk) >= 3 and len(".".join(chunk)) > char_limit:
            chunks.append(".".join(chunk).strip() + ".")
            chunk = chunk[-2:]
    if chunk is not None:
        chunks.append(".".join(chunk))
    return chunks
