import argparse
import json
import string
from nltk.stem import PorterStemmer

from inverted_index import *

















def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            keyword_search(args.query)

            
            pass
        case _:
            parser.print_help()


def keyword_search(searchTerm: str):
    with open("./data/movies.json","r") as f:
        movies = json.load(f)["movies"]
    n = 1


    punctDict = {}
    for c in string.punctuation:
        punctDict[c] = ""
    punctTransTable = str.maketrans(punctDict)

    stopWords = getStopwords()

    stemmer = PorterStemmer()

    searchTerm = searchTerm.lower().translate(punctTransTable)
    for movie in movies:
        title:str = movie["title"].lower().translate(punctTransTable)
        titleTokens = [x for x in title.split(" ") if x != ""]
        #print(titleTokens)
        searchTokens = [x for x in searchTerm.split(" ") if x != ""]
        searchTokens = [x for x in searchTokens if x not in stopWords]
        searchTokens = [stemmer.stem(token) for token in searchTokens]
        #print(searchTokens)
        match = False
        for sT in searchTokens:
            for tT in titleTokens:
                if sT in tT:
                    print(f"{n}. {movie["title"]}")
                    n += 1
                    match = True
                    break
                #if n >= 6:
                #    break
            if match:
                break
            #if n >= 6:
            #    break
        if n >= 6:
                break


def getStopwords()-> list[str]: 
    with open("./data/stopwords.txt","r") as f:
        content = f.read()
    lines = content.split("\n")
    return lines


if __name__ == "__main__":
    test_this()
    main()