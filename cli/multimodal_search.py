from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:


    def __init__(self,model_name="clip,ViT-B-32"):

        self.model = SentenceTransformer(model_name, device="cpu")


    def embed_image(self,image_path):
        image = Image.open(image_path)

        [embedding] = self.model.encode([image])
         
         return embedding
    


def verify_image_embedding(img_path):
   MS = MultimodalSearch()
   embedding = MS.embed_image(img_path)
   print(f"Embedding shape: {embedding.shape[0]} dimensions")