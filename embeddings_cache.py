import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path

CACHE_PATH = "corpus_embeddings.pkl"
INDEX_PATH = "corpus_faiss.index"

class EmbeddingCache:
    def __init__(self, corpus_texts: list[str]):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 6x faster than default
        self.corpus = corpus_texts
        self.index = None
        self._load_or_build()
    
    def _load_or_build(self):
        if Path(INDEX_PATH).exists() and Path(CACHE_PATH).exists():
            # Load cached index (instant startup after first run)
            self.index = faiss.read_index(INDEX_PATH)
            with open(CACHE_PATH, 'rb') as f:
                self.embeddings = pickle.load(f)
        else:
            # First time: compute and cache
            self.embeddings = self.model.encode(self.corpus, show_progress_bar=True)
            self.embeddings = np.array(self.embeddings).astype('float32')
            
            # Build FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine sim
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            
            # Save to disk
            faiss.write_index(self.index, INDEX_PATH)
            with open(CACHE_PATH, 'wb') as f:
                pickle.dump(self.embeddings, f)
    
    def search(self, query: str, top_k: int = 5) -> list[str]:
        # Encode query (~10ms)
        query_vec = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_vec)
        
        # FAISS search (~1ms for 5000 docs)
        scores, indices = self.index.search(query_vec, top_k)
        
        return [self.corpus[i] for i in indices[0]]