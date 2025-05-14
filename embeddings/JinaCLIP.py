from embeddings.BaseEmbeddingModel import BaseEmbeddingModel
from transformers import AutoModel, AutoProcessor
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

class JinaCLIP(BaseEmbeddingModel):
    def __init__(self, model_type="jinaai/jina-clip-v1", device="cuda"):
        """
        visit https://huggingface.co/jinaai for more models 
        """
        self.device = device if device else "cuda"
        self.model = AutoModel.from_pretrained(model_type, trust_remote_code=True).to("cuda").eval()
        self.processor = AutoProcessor.from_pretrained(model_type, trust_remote_code=True)
        self.embedding_dim = self.model.config.text_config.embed_dim
        
    def get_image_features(self, images):
        batch_size = len(images) if len(images) < 64 else 64
        return self.encode_images_in_batches(images, batch_size, self.processor, self.model)

    def get_text_features(self, texts):
        batch_size = len(texts) if len(texts) < 64 else 64
        return self.encode_texts_in_batches(texts, batch_size, self.processor, self.model)
    
    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def encode_images_in_batches(self,images, batch_size, processor, model):
        image_features = []
        num_batches = len(images) // batch_size + int(len(images) % batch_size > 0)
        
        if isinstance(images[0], str):
            images = [Image.open(image).convert("RGB") for image in images]
        
        for i in tqdm(range(0, len(images), batch_size), total=num_batches, desc="Encoding images"):
            batch_images = images[i:i+batch_size]
            inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(model.device)
            batch_features = model.get_image_features(**inputs)
            batch_features /= torch.norm(batch_features, dim=-1, keepdim=True)
            image_features.append(batch_features.cpu().float().numpy())
        
        return np.concatenate(image_features, axis=0)

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def encode_texts_in_batches(self,texts, batch_size, processor, model):
        text_features = []
        num_batches = len(texts) // batch_size + int(len(texts) % batch_size > 0)
        
        for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Encoding texts"):
            batch_texts = texts[i:i+batch_size]
            inputs = processor(batch_texts, return_tensors="pt", padding=True).to(model.device)
            batch_features = model.get_text_features(**inputs)
            batch_features /= torch.norm(batch_features, dim=-1, keepdim=True)
            text_features.append(batch_features.cpu().float().numpy())
        
        return np.concatenate(text_features, axis=0)