from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import re


# Load the model (downloads automatically the first time)
#model = SentenceTransformer('all-MiniLM-L6-v2')
#model
#print(f"Model loaded: {model}")
#print(f"Max sequence length: {model.max_seq_length}")

#model.encode(text)

class SemanticSearch:


    def __init__(self,model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")
        self.embeddings = None
        self.documents:list[dict]|None = None 
        self.document_map = {}

    def generate_embedding(self, text:str):
        if text.strip() == "":
            raise ValueError("Input text must not be empty")
        embedding = self.model.encode(text,device="cpu")
        return embedding
    
    def search(self,query:str,limit:int = 5):
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        qu_embedding= self.generate_embedding(query)
        similarities:list[tuple[float,dict]] = []
        for i, embedding in enumerate(self.embeddings):
            score = cosine_similarity(qu_embedding,embedding)
            doc = self.documents[i]
            similarities.append((score,doc))
        soreted_similarities =sorted(similarities,key = lambda x: x[0],reverse=True)
        results = []
        for i in range(limit):
            res = {}
            res["score"] = soreted_similarities[i][0]
            res["title"] = soreted_similarities[i][1]["title"]
            res["description"] = soreted_similarities[i][1]["description"]
            results.append(res)
        return results


    
    def build_embeddings(self,documents: list[dict]): 
        self.documents = documents 
        mov_strings = []
        for doc in documents:
            self.document_map[doc["id"]]=doc 
            mov_strings.append(f"{doc['title']}: {doc['description']}")

        
        self.embeddings = self.model.encode(mov_strings,show_progress_bar=True,device="cpu")

        with open("cache/movie_embeddings.npy","wb") as f:
            np.save(f,self.embeddings)

        return self.embeddings
    
    def load_or_create_embeddings(self,documents):
        self.documents = documents 
        mov_strings = []
        for doc in documents:
            self.document_map[doc["id"]]=doc 
            mov_strings.append(f"{doc['title']}: {doc['description']}")
        
        if os.path.exists("cache/movie_embeddings.npy"):
            with open("cache/movie_embeddings.npy","rb") as f:
                self.embeddings = np.load(f)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        
        self.embeddings = self.build_embeddings(documents)
        return self.embeddings
        








############| Other Functions |#####################

def cosine_similarity(vec1,vec2):
    dot_product = np.dot(vec1,vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def load_movies():
    with open("./data/movies.json","r") as f:
        documents: list[dict] = json.load(f)["movies"]
    return documents

        
############| Command functions |###################

def verify_model():
    model = 'all-MiniLM-L6-v2'
    SS = SemanticSearch()
    mod = str(SS.model)
    print(f"Model loaded: {mod}")
    print(f"Max sequence length: {SS.model.max_seq_length}")

def embed_text(text: str):
    
    SS = SemanticSearch()
    embedding = SS.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    SS = SemanticSearch()
    with open("./data/movies.json","r") as f:
        documents: list[dict] = json.load(f)["movies"]
    embeddings = SS.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query:str):
    SS = SemanticSearch()
    embedding = SS.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")



###########| Child Class: Chunked Semantic Search |#################



class ChunkedSemanticSearch(SemanticSearch):

    def __init__(self,model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings:list = None 
        self.chunk_metadata:list = None


    def search_chunks(self,query:str,limit: int = 10):
        query_embedding = self.generate_embedding(query)

        chunk_scores:list[dict] = []
        for i,ce in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding,ce)
            metadata = self.chunk_metadata["chunks"][i]
            chunk_scores.append({
                "chunk_idx":metadata["chunk_idx"],
                "movie_idx":metadata["movie_idx"],
                "score": score
            })

        score_map = {}
        for cs in chunk_scores:
            movie_idx = cs["movie_idx"]
            if movie_idx not in score_map:
                score_map[movie_idx] = cs["score"]
            if score_map[movie_idx] < cs["score"]:
                score_map[movie_idx] = cs["score"]
            
        soreted_movies = sorted(score_map.items(),key=lambda x: x[1],reverse=True)[0:limit]
        
        results = []
        for movie_idx,score in soreted_movies:
            doc:dict = self.documents[movie_idx]
            res = {
                "id":self.documents[movie_idx]["id"],
                "title": doc["title"],
                "document":doc["description"][:100],
                "score": score# round(score_map[id],5),
                #"metadata":{} #Oh my f god, why make me do all the previous steps like that if you wanted me to put the meta data in here afterwards?
                } # If this result needs metadata, I first need a metadata map that keeps it mapped to film ids, on god.
            results.append(res)
        return results



        

    def build_chunk_embeddings(self,documents:list[dict]):
        self.document_map = {}
        self.documents = documents 
        mov_strings = []
        for doc in documents:
            self.document_map[doc["id"]]=doc 
            #mov_strings.append(f"{doc['title']}: {doc['description']}")

        all_chunks:list[str] = []
        metadata: list[dict] = []
        for i,doc in enumerate(documents):
            description = doc.get("description")
            if description == "":
                continue 
            chunks=perform_semantic_chunking(description,4,1)
            #all_chunks += chunks

            for j,chunk in enumerate(chunks):
                all_chunks.append(chunk)
                meta = {
                    "movie_idx":i,
                    "chunk_idx":j,
                    "total_chunks":len(chunks)
                    }
                metadata.append(meta)

        
        self.chunk_embeddings = self.model.encode(all_chunks,show_progress_bar=True)
        self.chunk_metadata = metadata



        #self.embeddings = self.model.encode(mov_strings,show_progress_bar=True)

        with open("cache/chunk_embeddings.npy","wb") as f:
            np.save(f,self.chunk_embeddings)

        with open("cache/chunk_metadata.json","w") as f:
            json.dump({"chunks":metadata,"total_chunks":len(all_chunks)},f,indent=2)
            

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self,documents:list[dict]) -> np.ndarray:
        self.document_map = {}
        self.documents = documents 
        
        for doc in documents:
            self.document_map[doc["id"]]=doc 

        if not os.path.exists("cache/chunk_embeddings.npy") or not "chunk_metadata.json.npy":
            return self.build_chunk_embeddings(documents=documents)
        
        with open("cache/chunk_embeddings.npy","rb") as f:
            self.chunk_embeddings= np.load(f)

        with open("cache/chunk_metadata.json","r") as f:
            self.chunk_metadata = json.load(f)

        #print(self.chunk_metadata)

        if not len(self.chunk_embeddings) == len(self.chunk_metadata["chunks"]):
            print("Big problem! Something is wrong with embeddings or metadata, they don't have the same length.")

        return self.chunk_embeddings
            


############| Chunked Embedding assistance functions |################

def perform_semantic_chunking(text:str,chunk_size:int,overlap:int = 0):
    text = text.strip()
    if not text:
        return []
    
    sentences = re.split(pattern= r"(?<=[.!?])\s+",string=text)
    if len(sentences) == 1 and sentences[0][-1] not in ".!?":
        return sentences
    
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences is []:
        return []
            
    chunks = []
    while len(sentences) > chunk_size:
        chunks.append(" ".join(sentences[0:chunk_size]))
        sentences = sentences[chunk_size-overlap:]
    chunks.append(" ".join(sentences))



    return chunks