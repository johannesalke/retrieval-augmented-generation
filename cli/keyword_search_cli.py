import argparse
import json
import string
from nltk.stem import PorterStemmer
import pickle
import os

#from inverted_index import *

















def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    build_parser = subparsers.add_parser("build", help="Build an inverted index")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            keyword_search(args.query)

            
            pass
        case "build":
            ii =InvertedIndex(Tokenizer())
            ii.build(getMovies())
            ii.save()
            docs =ii.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")

            
            pass
        case _:
            parser.print_help()


def keyword_search(searchTerm: str):
    movies = getMovies()
    n = 1

    tokenizer = Tokenizer()
    searchTokens = tokenizer.tokenize(searchTerm)

    
    for movie in movies:
        titleTokens = tokenizer.tokenize(movie["title"])
        #print(titleTokens)
        
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



class Tokenizer: #Centralizing the logic so it can be reused for reverse indexing without the overhead of having to recreate the stemmer, stopword list and punctuation map every single time. 

    def __init__(self):
        punctDict = {}
        for c in string.punctuation:
            punctDict[c] = ""
        self.punctTransTable = str.maketrans(punctDict) #Translation table for removing punctuation

        self.stopWords = getStopwords() #Get low-value stop words to be deleted

        self.stemmer = PorterStemmer() #Get a stemmer to par words down to their root

    def tokenize(self,text:str) -> list[str]:
        text = text.lower().translate(self.punctTransTable)

        searchTokens = [x for x in text.split(" ") if x != ""]
        searchTokens = [x for x in searchTokens if x not in self.stopWords]
        searchTokens = [self.stemmer.stem(token) for token in searchTokens]
        return searchTokens


class InvertedIndex:
    index: dict[str,set[int]] = {}# Maps tokens to sets of DocumentIDs
    docmap: dict[int,dict] = {}
    tokenizer: Tokenizer

    def __init__(self,tokenizer: Tokenizer):
        self.tokenizer = tokenizer


    def __add_documents(self,doc_id: int,text:str):
        #1. Tokenize text 
        tokens = self.tokenizer.tokenize(text)

        for token in tokens:
            if self.index.get(token) == None:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self,term: str) -> list[int]:
        ids = self.index[term.lower()]
        
        return sorted(ids)
        print("test")

    def build(self,movies: list[dict]):
        for m in movies:
            self.__add_documents(int(m["id"]),f"{m['title']} {m['description']}")
            self.docmap[m["id"]] = m
        
    def save(self):
        if not os.path.exists("cache"):
            os.makedirs("cache")
        with open("cache/index.pkl","wb") as idx:
            pickle.dump(self.index,idx)
        with open("cache/docmap.pkl","wb") as docmap:
            pickle.dump(self.index,docmap)

    def load(self):
        if not os.path.exists("cache/index.pkl") or not os.path.exists("cache/docmap.pkl"):
            print("Error: Cached index files not found on disk")
            exit(1)
            return 
        with open("cache/index.pkl","rb") as idx:
            self.index= pickle.load(idx)
        with open("cache/docmap.pkl","rb") as docmap:
            self.docmap= pickle.load(docmap)
        
        

        



    




def getStopwords()-> list[str]: 
    with open("./data/stopwords.txt","r") as f:
        content = f.read()
    lines = content.split("\n")
    return lines

def getMovies() -> list[dict]:
    with open("./data/movies.json","r") as f:
        movies = json.load(f)["movies"]
    return movies



if __name__ == "__main__":
    
    
    main()


