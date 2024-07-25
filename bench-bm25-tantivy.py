from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import re
import json
import os
import time
import lance
from tqdm import tqdm
import tantivy

DATASET = os.getenv("DATASET", "quora")


def load_queries():
    queries = {}

    with open(f"data/{DATASET}/queries.jsonl", "r") as file:
        for line in file:
            row = json.loads(line)
            queries[row["_id"]] = {**row, "doc_ids": []}

    with open(f"data/{DATASET}/qrels/test.tsv", "r") as file:
        next(file)
        for line in file:
            query_id, doc_id, score = line.strip().split("\t")
            if int(score) > 0:
                queries[query_id]["doc_ids"].append(doc_id)

    queries_filtered = {}
    for query_id, query in queries.items():
        if len(query["doc_ids"]) > 0:
            queries_filtered[query_id] = query

    return queries_filtered


def sanitize_query_for_tantivy(query):
    # escape special characters
    query = re.sub(r'([+\-!(){}\[\]^"~*?:\\<])', r" ", query)
    return query


def main():
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("body", stored=True, tokenizer_name="en_stem")
    schema_builder.add_text_field("doc_id", stored=False)
    schema = schema_builder.build()
    index = tantivy.Index(schema, path=f"data/{DATASET}/bm25.tantivy/")

    searcher = index.searcher()

    number_of_queries = 10_000
    queries = list(load_queries().values())[:number_of_queries]
    queries = [sanitize_query_for_tantivy(query["text"]) for query in queries]

    for concurrency in [4, 8, 16]:
        latencies = []
        limit = 10

        def search_bm25(query):
            start = time.time()
            query = index.parse_query(query, ["body"])
            hits = searcher.search(query, limit).hits
            docs = [searcher.doc(doc_address) for (_, doc_address) in hits]
            latencies.append(time.time() - start)

            return hits

        thread_pool = ThreadPoolExecutor(max_workers=concurrency)
        start = time.time()
        results = thread_pool.map(
            search_bm25, queries, chunksize=number_of_queries // concurrency / 4
        )
        for _ in tqdm(results, total=number_of_queries):
            pass
        latencies.sort()
        mean = sum(latencies) / len(latencies)
        p50 = latencies[int(len(latencies) * 0.5) - 1]
        p90 = latencies[int(len(latencies) * 0.9) - 1]
        p99 = latencies[int(len(latencies) * 0.99) - 1]
        print(
            f"concurrency: {concurrency}, QPS: {number_of_queries/ (time.time() - start)}, mean: {mean}, p50: {p50}, p90: {p90}, p99: {p99}"
        )


if __name__ == "__main__":
    main()
