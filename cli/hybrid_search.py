import os

from keyword_search_cli import InvertedIndex,Tokenizer,getMovies
from lib.semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex(Tokenizer())
        if not os.path.exists("cache/index.pkl"):
            self.idx.build(getMovies())
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha=0.5, limit=5):
        bm25_results = self._bm25_search(query,limit*500)
        semantic_results = self.semantic_search.search_chunks(query,limit*500)

        self.scores_map = {}

        

        bm25_scores:list[float] = [x["score"] for x in bm25_results] 
        bm25_norm_scores = normalize_scores(bm25_scores)
        for i,_ in enumerate(bm25_results):
            bm25_results[i]["score"] = bm25_norm_scores[i]
            idx:int = bm25_results[i]["movie"]["id"]
            if not self.scores_map.get(idx):
                self.scores_map[idx] = {}
            self.scores_map[idx]["keyword_score"] = bm25_norm_scores[i]
        
        #print(f"After BM loop: {idx} (type: {type(idx)})")
        semantic_scores:list[float] = [x["score"] for x in semantic_results] 
        #print(semantic_scores)
        semantic_norm_scores = normalize_scores(semantic_scores)
        for i,_ in enumerate(semantic_scores):
            semantic_results[i]["score"] = semantic_norm_scores[i]
            idx:int = semantic_results[i]["id"]
            #print(f"Looking up after idx reassignment: {idx} (type: {type(idx)})")
            
            if not self.scores_map.get(idx):
                self.scores_map[idx] = {}
            self.scores_map[idx]["semantic_score"] = semantic_norm_scores[i]

        
        #print(list(self.idx.docmap.keys())[:5])
        #print(f"Looking up: {idx} (type: {type(idx)})")
        #print(self.idx.docmap)
        for idx in self.scores_map:
            
            self.scores_map[idx]["document"] = self.idx.docmap[idx]
            key_score = self.scores_map[idx].get("keyword_score")
            semantic_score = self.scores_map[idx].get("semantic_score")
            if semantic_score is None:
                semantic_score = 0.0
            if key_score is None: 
                key_score = 0.0
            hybrid_score = (key_score*alpha+semantic_score*(1-alpha))
            self.scores_map[idx]["hybrid_score"]=hybrid_score

        sorted_results = sorted(self.scores_map.values(),key= lambda x: x["hybrid_score"],reverse=True)
        return sorted_results[:limit]
    

    def rrf_search(self,query,k=60,limit=10):
        bm25_results =self._bm25_search(query,limit*500)
        semantic_results =self.semantic_search.search_chunks(query,limit*500)

        self.rank_map = {}

        
        for i,_ in enumerate(bm25_results):
            
            idx:int = bm25_results[i]["movie"]["id"]
            if not self.rank_map.get(idx):
                self.rank_map[idx] = {}
            self.rank_map[idx]["bm25_rank"] = i+1
            #self.scores_map[idx]["document"] = res["movie"]

        
        
        for i,_ in enumerate(semantic_results):
            idx:int = semantic_results[i]["id"]
           
            if not self.rank_map.get(idx):
                self.rank_map[idx] = {}
            self.rank_map[idx]["semantic_rank"] = i+1

            
        for idx in self.rank_map:
            self.rank_map[idx]["rff_score"] = 0.0

            bm25_rank = self.rank_map[idx].get("bm25_rank")
            if bm25_rank:
                self.rank_map[idx]["rff_score"] += rrf_score(bm25_rank,k)

            semantic_rank = self.rank_map[idx].get("semantic_rank")
            if semantic_rank:
                self.rank_map[idx]["rff_score"] += rrf_score(semantic_rank,k)
            
            self.rank_map[idx]["document"] = self.idx.docmap[idx]

        sorted_results = sorted(self.rank_map.values(),key= lambda x: x["rff_score"],reverse=True)
        return sorted_results[:limit]

        
        

        

    
    
    





def normalize_scores(scores: list[float], key = lambda x: x)->list[float]:
    smin = min(scores,key = lambda x: key(x))
    smax = max(scores,key = lambda x: key(x))
    diff = smax-smin 
    if diff == 0:
        return [1.0]*len(scores)

    normalized_scores = [(s-smin)/diff for s in scores]
    return normalized_scores

def rrf_score(rank,k=60):
    return 1/(k+rank)