import json
from pathlib import Path
from typing import Dict, List

from langchain.docstore.document import Document


def write_documents_to_json(docs: List[Document], json_path):
    split_json = [
        {"page_content": d.page_content, "metadata": d.metadata} for d in docs
    ]
    with open(json_path, "w") as f:
        json.dump(split_json, f)


def json_to_doc(dataset_path):
    data: List[Dict] = json.loads(Path(dataset_path).read_text())
    docs = [
        Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data
    ]
    return docs
