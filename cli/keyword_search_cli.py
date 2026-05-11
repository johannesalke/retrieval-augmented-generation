import argparse
import json
import string
from nltk.stem import PorterStemmer
import pickle
import os
import collections
import math

#from inverted_index import *




BM25_K1 = 1.5
BM25_B = 0.75
CACHE = "cache"











def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build an inverted index")

    tf_parser = subparsers.add_parser("tf", help="Returns the term frequency of movie object")
    tf_parser.add_argument("doc_id", type=int, help="ID of movie to analyse")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser("idf", help="Checks the inverse document frequency of a term")
    idf_parser.add_argument("term", type=str, help="Term to get the idf for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Returns the TF-IDF score of a movie")
    tfidf_parser.add_argument("doc_id", type=int, help="ID of movie to analyze")
    tfidf_parser.add_argument("term", type=str, help="Term to get score")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    
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
            
            pass
        case "tf":
            print(f"Printing term frequency for movie with the id: {args.doc_id} and term: {args.term}")
            
            ii =InvertedIndex(Tokenizer())
            ii.load()
            n = ii.get_tf(args.doc_id,args.term)
            print("The frequency is: "+str(n))

            pass
        case "idf":
            term = args.term
            ii =InvertedIndex(Tokenizer())
            ii.load()
            [term] = ii.tokenizer.tokenize(term)
    
            term_match_doc_count = len(ii.get_documents(term))
            total_doc_count = len(ii.docmap)
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

            pass
        case "tfidf":
            ii =InvertedIndex(Tokenizer())
            ii.load()

            tf = ii.get_tf(args.doc_id,args.term)
    
            term_match_doc_count = len(ii.get_documents(args.term))
            total_doc_count = len(ii.docmap)
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

            pass

        case "bm25idf":
            ii =InvertedIndex(Tokenizer())
            ii.load()
            bm25idf = ii.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

            pass

        case "bm25tf":
            ii =InvertedIndex(Tokenizer())
            ii.load()
            bm25tf = ii.get_bm25_tf(args.doc_id,args.term,args.k1,args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

            pass

        case "bm25search":
            ii =InvertedIndex(Tokenizer())
            ii.load()
            
            results = ii.bm25_search(args.query,5)
            for n in range(5):
                result = results[n]
                movie = result["movie"]
                score = result["score"]

                print(f"{n+1}. ({movie["id"]}) {movie["title"]} - Score: {score:.2f}")
            pass



        
        case _:
            parser.print_help()


def keyword_search(searchTerm: str):
    movies = getMovies()
    n = 1

    tokenizer = Tokenizer()
    searchTokens = tokenizer.tokenize(searchTerm)

    II = InvertedIndex(tokenizer)
    II.load()
    #print(II.docmap)
    #with open("cache/doc_map.json","w") as f:
    #    json.dump(II.docmap,f)
    

    for token in searchTokens:
        doc_ids = II.get_documents(token)
        for id in doc_ids:
            #print(id)
            movie = II.docmap.get(id)
            print(f"{n}. {movie["title"]}")
            n += 1
            if n >= 6:
                return



class Tokenizer: #Centralizing the logic so it can be reused for reverse indexing without the overhead of having to recreate the stemmer, stopword list and punctuation map every single time. 
    
    def __init__(self):
        
        punctDict = {}
        for c in string.punctuation:
            punctDict[c] = ""
        self.punctTransTable = str.maketrans(punctDict) #Translation table for removing punctuation

        self.stopWords = getStopwords() #Get low-value stop words to be deleted

        self.stemmer = PorterStemmer() #Get a stemmer to par words down to their root
        self.index_path:str = "cache/index.pkl"


    def tokenize(self,text:str) -> list[str]:
        text = text.lower().translate(self.punctTransTable)
        text = " ".join(text.split("\n"))
        searchTokens = [x for x in text.split(" ") if x != ""]
        searchTokens = [x for x in searchTokens if x not in self.stopWords]
        searchTokens = [x for x in searchTokens if x not in self.stopWords]
        searchTokens = [self.stemmer.stem(token) for token in searchTokens]
        return searchTokens


class InvertedIndex:
    index: dict[str,set[int]] = {}# Maps tokens to sets of DocumentIDs
    docmap: dict[int,dict] = {}
    tokenizer: Tokenizer
    term_frequencies: dict[int, collections.Counter] = {}
    doc_lengths: dict[int,int] = {}
    avg_doc_length = 0
    
    

    def __init__(self,tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        
        


    def __add_documents(self,doc_id: int,text:str):
        #1. Tokenize text 
        tokens = self.tokenizer.tokenize(text)
        counter =collections.Counter()

        for token in tokens:
            if self.index.get(token) == None:
                self.index[token] = set()
            self.index[token].add(doc_id)
            counter[token] += 1

        self.term_frequencies[doc_id] = counter
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        total = 0
        for k in self.doc_lengths:
            total += self.doc_lengths[k]
        if len(self.doc_lengths) == 0:
            return 0.0
        return total/len(self.doc_lengths)
        


    def get_documents(self,term: str) -> list[int]:
        ids = self.index.get(term.lower())
        if ids == None:
            return []
        return sorted(ids)
    
    #def get_movie(self,id):
    #    if type(id) == "str":
    #        return self.docmap.get(int(id))

    def get_tf(self,doc_id:int,term:str):
        if len(term.split(" ")) >= 2:
            raise Exception("Error: This method should only be called with a single token :(")
        [term] = self.tokenizer.tokenize(term)
        counter = self.term_frequencies.get(doc_id)
        if counter == None:
            print("Error: non-existent document id")
            return 0
        n = counter.get(term)
        if n == None:
            n = 0
        return n
    
    def get_bm25_idf(self, term:str) -> float:
        if len(term.split(" ")) >= 2:
            raise Exception("Error: This method should only be called with a single token :(")
        [term] = self.tokenizer.tokenize(term)

        df = len(self.get_documents(term))
        N = len(self.docmap)

        return math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def get_bm25_tf(self,doc_id,term,k1=BM25_K1,b=BM25_B) -> float:
        tf:int= self.get_tf(doc_id,term) 
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.avg_doc_length)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 *length_norm)
        return bm25_tf
    
    def bm25(self,doc_id,term:str):
        bm25_tf = self.get_bm25_tf(doc_id,term)
        bm25_idf = self.get_bm25_idf(term)
        BM25 = bm25_tf * bm25_idf
        return BM25
    
    def bm25_search(self,query,limit:int = 5):
        query_tokens = self.tokenizer.tokenize(query)
        scores: dict[int,float] = {}
        #for idx in self.index:
        for term in query_tokens:
            doc_ids = self.get_documents(term)
            for id in doc_ids:
                current_score = scores.get(id)
                if current_score == None:
                    current_score = 0.0
                scores[id] = current_score + self.bm25(id,term)
        
        scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        #print(scores)
        n = 1
        results = []
        for key in scores.keys():
            results.append( {"movie": self.docmap[key],"score": scores[key]})
            n += 1
            if n == limit+1:
                break


        return results
    


    # Methods for building, saving and loading the II data

    def build(self,movies: list[dict]):
        for m in movies:
            self.__add_documents(int(m["id"]),f"{m['title']} {m['description']}")
            self.docmap[int(m["id"])] = m
        self.avg_doc_length = self.__get_avg_doc_length()
        
    def save(self):
        if not os.path.exists("cache"):
            os.makedirs("cache")
        with open("cache/index.pkl","wb") as idx:
            pickle.dump(self.index,idx)
        with open("cache/docmap.pkl","wb") as docmap:
            pickle.dump(self.docmap,docmap)
        with open("cache/term_frequencies.pkl","wb") as freq:
            pickle.dump(self.term_frequencies,freq)
        with open("cache/doc_lengths.pkl","wb") as lengths:
            pickle.dump(self.doc_lengths,lengths)

    

    def load(self):
        if not os.path.exists("cache/index.pkl") or not os.path.exists("cache/docmap.pkl"):
            print("Error: Cached index files not found on disk")
            exit(1)
            return 
        with open("cache/index.pkl","rb") as idx:
            self.index= pickle.load(idx)
        with open("cache/docmap.pkl","rb") as docmap:
            self.docmap= pickle.load(docmap)
        with open("cache/term_frequencies.pkl","rb") as freq:
            self.term_frequencies= pickle.load(freq)
        with open("cache/doc_lengths.pkl","rb") as lengths:
            self.doc_lengths= pickle.load(lengths)
        
        self.avg_doc_length = self.__get_avg_doc_length()
        
        

        
#def bm25_idf_command(term:str):



    




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


