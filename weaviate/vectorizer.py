import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel

class JobVectorizer:
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def vectorize_job(self, job_data: Dict[str, str]) -> np.ndarray:
        # Use the relevant fields for vectorization
        text = " ".join([
            job_data.get("title", ""), 
            job_data.get("company", ""), 
            job_data.get("description", ""), 
            job_data.get("place", ""), 
            job_data.get("seniority_level", ""), 
            job_data.get("employmnet_type", "")
        ])
        return self.embed_text(text)