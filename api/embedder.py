import hashlib
import re
import numpy as np


class RawFixedEmbedder:

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _embed_single(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        vector = np.zeros(self.dimensions, dtype=np.float32)

        for token in tokens:
            hash_val = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = hash_val % self.dimensions
            vector[idx] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()

    # Documents ke liye (LangChain Document objects & raw strings)
    def embed_documents(self, texts: list) -> list[list[float]]:
        cleaned_texts = []
        for item in texts:
            if hasattr(item, "page_content"):
                cleaned_texts.append(item.page_content)
            else:
                cleaned_texts.append(str(item))

        return [self._embed_single(t) for t in cleaned_texts]

    # Query Embeddings ke liye (FAISS isay internally call karta hai)
    def embed_query(self, text: str) -> list[float]:
        return self._embed_single(text)

    # Callable Guard (TypeError: 'RawFixedEmbedder' object is not callable fix)
    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)