import argparse
import hybrid_search as hs
from lib.semantic_search import load_movies
import lib.llm_functions as llm


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize search results")
    summarize_parser.add_argument("query", type=str, help="Query to be sought for")
    summarize_parser.add_argument("--limit", type=int,default=5, help="Number of results. Default 5")

    citations_parser = subparsers.add_parser("citations", help="Summarize search results")
    citations_parser.add_argument("query", type=str, help="Query to be sought for")
    citations_parser.add_argument("--limit", type=int,default=5, help="Number of results. Default 5")

    question_parser = subparsers.add_parser("question", help="Summarize search results")
    question_parser.add_argument("question", type=str, help="Query to be sought for")
    question_parser.add_argument("--limit", type=int,default=5, help="Number of results. Default 5")



    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            HS = hs.HybridSearch(load_movies())
            results = HS.rrf_search(query,60,5)
            doc_list = []
            for i,res in enumerate(results):
                doc = res["document"]
                doc_list.append(f"{i+1}. {doc.get('title', '')} - {doc.get('description', '')}")
            docs = chr(10).join(doc_list)
            rag_response = llm.rag(query,docs)

            print("Search Results:")
            for res in results:
                print(f"- {res["document"]["title"]}")
            print("RAG Response:")
            print(rag_response)

        case "summarize":
            query = args.query
            HS = hs.HybridSearch(load_movies())
            results = HS.rrf_search(query,60,5)
            doc_list = []
            for i,res in enumerate(results):
                doc = res["document"]
                doc_list.append(f"{i+1}. {doc.get('title', '')} - {doc.get('description', '')}")
            docs = chr(10).join(doc_list)
            summary_response = llm.summarize(query,docs)

            print("Search Results:")
            for res in results:
                print(f"- {res["document"]["title"]}")
            print("LLM Summary:")
            print(summary_response)

        case "citations":
            query = args.query
            HS = hs.HybridSearch(load_movies())
            results = HS.rrf_search(query,60,5)
            doc_list = []
            for i,res in enumerate(results):
                doc = res["document"]
                doc_list.append(f"{i+1}. {doc.get('title', '')} - {doc.get('description', '')}")
            docs = chr(10).join(doc_list)
            summary_response = llm.citations(query,docs)

            print("Search Results:")
            for res in results:
                print(f"- {res["document"]["title"]}")
            print("LLM Summary:")
            print(summary_response)

        case "question":
            question = args.question
            HS = hs.HybridSearch(load_movies())
            results = HS.rrf_search(question,60,5)
            doc_list = []
            for i,res in enumerate(results):
                doc = res["document"]
                doc_list.append(f"{i+1}. {doc.get('title', '')} - {doc.get('description', '')}")
            docs = chr(10).join(doc_list)
            question_response = llm.question(question,docs)

            print("Search Results:")
            for res in results:
                print(f"- {res["document"]["title"]}")
            print("Answer:")
            print(question_response)


        case _:
            parser.print_help()

if __name__ == "__main__":
    main()


