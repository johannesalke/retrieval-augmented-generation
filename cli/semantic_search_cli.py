#!/usr/bin/env python3

import argparse
import lib.semantic_search as semantic_search
import numpy as np
import re




def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the thing is working")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed some text via the model")
    embed_text_parser.add_argument("text", type=str, help="Text to be embedded")

    verif_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Vereify the damn embeddings")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed a query via the model")
    embed_query_parser.add_argument("query", type=str, help="Query to be embedded")

    search_parser = subparsers.add_parser("search", help="Search for a query via semantic search")
    search_parser.add_argument("query", type=str, help="Query to be sought for")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Top number of results. Default: 5")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a query with simple chunking")
    chunk_parser.add_argument("query", type=str, help="Query to be sought for")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=200, help="Size of chunks in words. Default: 200")
    chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="Degree of overlap in words. Default: 0")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk a query with semantic chunking")
    semantic_chunk_parser.add_argument("query", type=str, help="Query to be sought for")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs='?', default=4, help="Max size of chunks in words. Default: 4")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="Degree of overlap in words. Default: 0")

    build_chunk_embeddings_parser = subparsers.add_parser("build_chunk_embeddings", help="Build chunk embeddings for the first time. May take a while...")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Load chunk embeddings, or build them if they don't already exist")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search with a chunked semantic search methodology")
    search_chunked_parser.add_argument("query", type=str, help="Query to be sought for")
    search_chunked_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results to display. Default: 5")
        

    args = parser.parse_args()

    match args.command:
        case "verify":
            semantic_search.verify_model()

        case "embed_text":
            semantic_search.embed_text(args.text)
        
        case "verify_embeddings":
            semantic_search.verify_embeddings()

        case "embed_query":
            semantic_search.embed_query_text(args.query)

        case "search":
            SS = semantic_search.SemanticSearch()
            documents = semantic_search.load_movies()
            SS.load_or_create_embeddings(documents)
            results = SS.search(args.query,args.limit)
            for i,res in enumerate(results):
                print(f"{i}. {res["title"]} (score: {res["score"]})\n{res["description"][0:50]}")

        case "chunk":
            words = args.query.split(" ")
            chunk_size:int = args.chunk_size
            chunks = []
            overlap = args.overlap
            while len(words) > chunk_size:
                chunks.append(" ".join(words[0:chunk_size]))
                words = words[chunk_size-overlap:]
            chunks.append(" ".join(words))
            print(f"Chunking {len(args.query)} characters")
            for i,chnk in enumerate(chunks):
                print(f"{i+1}. {chnk}")

        case "semantic_chunk":
            chunks = semantic_search.perform_semantic_chunking(args.query,args.max_chunk_size,overlap=args.overlap)
            #sentences = re.split(pattern= r"(?<=[.!?])\s+",string=args.query)
            #chunk_size:int = args.max_chunk_size
            #chunks = []
            #while len(sentences) > chunk_size:
            #    chunks.append(" ".join(sentences[0:chunk_size]))
            #    sentences = sentences[chunk_size-args.overlap:]
            #chunks.append(" ".join(sentences))
            #print(f"Semantically chunking {len(args.query)} characters")
            if chunks == []:
                print("Nothing!")
            for i,chnk in enumerate(chunks):
                print(f"{i+1}. {chnk}")

        case "build_chunk_embeddings":
            CSS = semantic_search.ChunkedSemanticSearch()
            CSS.build_chunk_embeddings(semantic_search.load_movies())
            print("Build successfull")

        case "embed_chunks":
            CSS = semantic_search.ChunkedSemanticSearch()
            embeddings = CSS.load_or_create_chunk_embeddings(semantic_search.load_movies())
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            CSS = semantic_search.ChunkedSemanticSearch()
            CSS.load_or_create_chunk_embeddings(semantic_search.load_movies())
            results = CSS.search_chunks(args.query,limit = args.limit)
            print(f"Displaying top {args.limit} results:")
            for i,res in enumerate(results):
                print(f"{i+1}. {res["title"]} (score: {res["score"]:.4f})")
                #print(f"   {res["document"]}...")


        case _:
            parser.print_help()

            

























if __name__ == "__main__":
    main()