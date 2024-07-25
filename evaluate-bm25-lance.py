import re
import json
import os
import lance
from tqdm import tqdm

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
    dataset = lance.dataset(f"data/{DATASET}/bm25.lance")

    def search_bm25(query, limit):
        query = sanitize_query_for_tantivy(query)
        plan = dataset.scanner(
            columns=["doc_id"],
            full_text_query={
                "query": query,
                "columns": ["doc_text"],
            },
            limit=limit,
        )
        table = plan.to_table()
        hits = table.to_pylist()
        return hits

    n = 0
    hits = 0
    limit = 10
    number_of_queries = 100_000

    queries = load_queries()

    num_queries = 0
    num_responses = 0

    recalls = []
    precisions = []

    for idx, query in tqdm(enumerate(queries.values())):
        if idx >= number_of_queries:
            break

        num_queries += 1

        result = search_bm25(query["text"], limit)
        # print(f"Processing query: {query}, hits: {len(result)}")
        num_responses += len(result)

        found_ids = []

        for hit in result:
            found_ids.append(hit["doc_id"])

        query_hits = 0
        for doc_id in query["doc_ids"]:
            n += 1
            if doc_id in found_ids:
                hits += 1
                query_hits += 1

        recall = query_hits / len(query["doc_ids"])
        recalls.append(recall)
        precisions.append(query_hits / limit)

        if idx % 200 == 0:
            average_recall = sum(recalls) / len(recalls)
            print(f"Processing query: {query}, recall: {average_recall}")

    print(f"Total hits: {hits} out of {n}, which is {hits/n}")

    print(f"Precision: {hits/(num_queries * limit)}")

    average_precision = sum(precisions) / len(precisions)

    print(f"Average precision: {average_precision}")

    average_recall = sum(recalls) / len(recalls)

    print(f"Average recall: {average_recall}")


if __name__ == "__main__":
    main()
