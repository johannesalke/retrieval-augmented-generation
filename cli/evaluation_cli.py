import argparse
import json
import hybrid_search as hs
import lib.semantic_search as ss

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    golden = load_golden_dataset()["test_cases"]
    HS = hs.HybridSearch(ss.load_movies())
    print(f"k={limit}\n")
    for g in golden:
        results = HS.rrf_search(g["query"],k=60,limit=limit)
        relevant_docs = g["relevant_docs"]
        result_titles = [res["document"]["title"] for res in results]

        matches = 0
        for title in result_titles:
            if title in relevant_docs:
                matches += 1
        precision = matches/limit

        matches = 0
        for title in relevant_docs:
            if title in result_titles:
                matches += 1
        recall = matches/len(relevant_docs)

        if precision == recall == 0:
            f1 = 0
        else:
            f1 = 2*(precision * recall) / (precision + recall)

        retrieved = [res["document"]["title"] for res in results]
        
        print(f"- Query: {g["query"]}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {", ".join(retrieved)}")
        print(f"  - Relevant: {", ".join(relevant_docs)}\n")







    








def load_golden_dataset():
    with open("data/golden_dataset.json","r") as f:
        golden = json.load(f)
        print(type(golden))
    return golden


if __name__ == "__main__":
    main()