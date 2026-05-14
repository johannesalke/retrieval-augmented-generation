import argparse
import hybrid_search as hs
from lib.semantic_search import load_movies
import lib.llm_functions as llm
import time
import json
from sentence_transformers import CrossEncoder

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    normalize_parser= subparsers.add_parser("normalize", help="Normalize a list of commands")
    normalize_parser.add_argument("scores", type=float, nargs='*', default=0, help="List of scores to normalize")

    weighted_search_parser= subparsers.add_parser("weighted-search", help="Hybrid search weighted towards semantic of keyword search")
    weighted_search_parser.add_argument("query", type=str, help="Query to be sought for")
    weighted_search_parser.add_argument("--alpha", type=float,default=0.5, help="Weight. Higher leans more towards keywords. Default 0.5")
    weighted_search_parser.add_argument("--limit", type=int,default=5, help="Number of results. Default 5")

    rrf_search_parser= subparsers.add_parser("rrf-search", help="Reciprocal Rank Fusion Search")
    rrf_search_parser.add_argument("query", type=str, help="Query to be sought for")
    rrf_search_parser.add_argument("--k", type=int,default=60, help="Weight parameter. Higher k means less difference between high and low ranks Default: 60")
    rrf_search_parser.add_argument("--limit", type=int,default=5, help="Number of results. Default 5")
    rrf_search_parser.add_argument("--enhance",type=str,choices=["spell","rewrite","expand"],help="LLM prompt enhancement type")
    rrf_search_parser.add_argument("--rerank-method",type=str,choices=["individual","batch","cross_encoder"],help="LLM reranking method")
    rrf_search_parser.add_argument("--evaluate",action="store_true",help="Have an LLM evaluate the quality of the matches.")


    args = parser.parse_args()

    match args.command:

        case "normalize":
            scores = args.scores
            if scores == []:
                return
            norm_scores = hs.normalize_scores(scores)
            for score in norm_scores:
                print(f"* {score:.4f}")

        case "weighted-search":
            HS = hs.HybridSearch(load_movies())
            results = HS.weighted_search(args.query,args.alpha,args.limit)
            for i,res in enumerate(results):
                title = res["document"]["title"]
                description = res["document"]["description"][:100]
                print(f"{i+1}. {title}")
                print(f"Hybrid Score: {res["hybrid_score"]:.4f}")
                print(f"BM25: {res["keyword_score"]:.4f}, Semantic: {res["semantic_score"]:.4f}")
                print(description)

        case "rrf-search":
            HS = hs.HybridSearch(load_movies())
            query = args.query
            enhance = args.enhance
            limit = args.limit
            rerank_method = args.rerank_method
            if rerank_method != "":
                limit = args.limit*5



            if enhance == "spell":
                enhanced_query = llm.enhance_query_spelling(args.query)
                print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
                query = enhanced_query
            if enhance == "rewrite":
                enhanced_query = llm.enhance_query_rewrite(args.query)
                print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
                query = enhanced_query
            if enhance == "expand":
                enhanced_query = llm.enhance_query_expand(args.query)
                print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
                query = enhanced_query

            results = HS.rrf_search(args.query,args.k,limit)

            if rerank_method == "individual":
                for i,res in enumerate(results):
                    llm_rank = llm.rerank_individual(query,res["document"])
                    res["rerank_score"] = int(llm_rank)
                    time.sleep(10)
                    print(res["rerank_score"])
                    results[i] = res
                results = sorted(results,key=lambda x: x["rerank_score"],reverse=True)

            if rerank_method == "batch":
                res_list = [x["document"] for x in results]
                doc_list_str = json.dumps(res_list)
                llm_batch_ranks_text = llm.rerank_batch(query,doc_list_str)
                llm_ranks:list = json.loads(llm_batch_ranks_text)
                print(llm_ranks)
                if len(llm_ranks) != len(results):
                    raise Exception("The batch reranking response was flawed: Wrong length")
                for res in results:
                    for i,rr in enumerate(llm_ranks):
                        if rr == res["document"]["id"]:
                            res["rerank_rank"] = i
                            break
                results = sorted(results,key=lambda x: x["rerank_rank"],reverse=True)

            if rerank_method == "cross_encoder":
                pairs = []
                for res in results:
                    doc = res["document"]
                    pairs.append([query, f"{doc.get('title', '')} - {doc.get('description', '')}"])

                CE = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                scores = CE.predict(pairs)
                for i in range(len(results)):
                    results[i]["cross_encoder_score"] = scores[i]
                results = sorted(results,key=lambda x: x["cross_encoder_score"],reverse=True)


            for i,res in enumerate(results[:args.limit]):
                title = res["document"]["title"]
                description = res["document"]["description"][:100]
                bm25_rank = res.get("bm25_rank","")
                sem_rank = res.get("semantic_rank","")
                
                print(f"{i+1}. {title}")
                if rerank_method == "individual":
                    rerank = res["rerank_score"]
                    print(f" Re-rank Score: {rerank}/10")
                elif rerank_method == "batch":
                    rerank = res["rerank_rank"]
                    print(f" Re-rank Rank: {rerank}")
                elif rerank_method == "cross_encoder":
                    cross = res["cross_encoder_score"]
                    print(f" Cross Encoder Score: {cross:.3f}")


                print(f" RRF Score: {res["rff_score"]:.3f}")
                if not bm25_rank:
                    print(f" Semantic Rank: {sem_rank:d}")
                elif not sem_rank:
                    print(f" BM25 Rank: {bm25_rank:d}")
                else:
                    print(f" BM25 Rank: {bm25_rank:d}, Semantic Rank: {sem_rank:d}")
                print(" "+description)

            if args.evaluate == True:
                formatted_results = [f"{i+1}. Title: {res["document"]["title"]}, Description: {res["document"]["description"]}" for i,res in enumerate(results)]

                evaluations = json.loads(llm.evaluate_results(query=query,formatted_results=formatted_results))
                print("\n Final Evaluation")
                for i,eval in enumerate(evaluations):
                    print(f"{i+1}. {results[i]["document"]["title"]}: {eval}/3")


            

        case _:
            parser.print_help()



import os
from dotenv import load_dotenv


from google import genai





def enhance_query_spelling(query):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    prompt =  f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
    client = genai.Client(api_key=api_key)
    gen_res = client.models.generate_content(model="gemma-4-31b-it",contents=prompt)
    return gen_res.text

def enhance_query_rewrite(query):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    prompt =  f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
    client = genai.Client(api_key=api_key)
    gen_res = client.models.generate_content(model="gemma-4-31b-it",contents=prompt)
    return gen_res.text


def enhance_query_expand(query):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    prompt =  f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""
    client = genai.Client(api_key=api_key)
    gen_res = client.models.generate_content(model="gemma-4-31b-it",contents=prompt)
    return gen_res.text






if __name__ == "__main__":
    main()