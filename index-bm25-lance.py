import os
import json
from tqdm import tqdm
import lance
from typing import Iterable, List
import shutil
import pyarrow as pa


DATASET = os.getenv("DATASET", "quora")


def read_file(file_name: str) -> Iterable[str]:
    with open(file_name, "r") as file:
        for line in file:
            row = json.loads(line)
            yield row["_id"], row["text"]


def main():
    file_name = f"data/{DATASET}/corpus.jsonl"  # DATASET collection
    file_out = f"data/{DATASET}/bm25.lance"  # output file

    if os.path.exists(file_out):
        # remove direcotry recursively
        shutil.rmtree(file_out)

    if not os.path.exists(file_out):
        os.makedirs(file_out, exist_ok=True)

    doc_id_list = []
    doc_text_list = []
    for idx, (doc_id, doc_text) in enumerate(read_file(file_name)):
        doc_id_list.append(doc_id)
        doc_text_list.append(doc_text)
    doc_id = pa.array(doc_id_list)
    doc_text = pa.array(doc_text_list, type=pa.string())
    table = pa.table({"doc_id": doc_id, "doc_text": doc_text})
    dataset = lance.write_dataset(
        table,
        file_out,
    )
    # dataset = lance.dataset('gs://yang-bench/bm25.lance')
    print("created dataset, num_rows:", dataset.count_rows())

    dataset.create_scalar_index("doc_text", index_type="INVERTED")
    print("indexed")


if __name__ == "__main__":
    main()
