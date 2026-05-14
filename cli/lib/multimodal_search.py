from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.semantic_search import cosine_similarity,load_movies
import numpy as np

class MultimodalSearch:


    def __init__(self,doc_list=load_movies(),model_name="clip-ViT-B-32"):

        self.model = SentenceTransformer(model_name, device="cpu")
        self.documents = doc_list
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = self.model.encode(self.texts,show_progress_bar=True)
        self.save_text_embeddings()


    def embed_image(self,image_path):
        image = Image.open(image_path)

        [embedding] = self.model.encode([image])
         
        return embedding
    
    def search_with_image(self,image_path):
        img_embedding = self.embed_image(image_path)
        results = []
        for i,txt_embedding in enumerate(self.text_embeddings):
            sim = cosine_similarity(img_embedding,txt_embedding)
            res = {}
            res.update(self.documents[i])
            res["similarity_score"] = sim
            results.append(res)

        results = sorted(results,key= lambda res: res["similarity_score"],reverse=True)
        return results
    
    def save_text_embeddings(self):

         with open("cache/multimodal_text_embeddings.npy","wb") as f:
            np.save(f,self.text_embeddings)

            

    


def verify_image_embedding(img_path):
   MS = MultimodalSearch(load_movies())
   embedding = MS.embed_image(img_path)
   print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path):
    MS = MultimodalSearch(load_movies())
    results = MS.search_with_image(image_path)

    return results