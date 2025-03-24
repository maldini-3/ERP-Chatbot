import faiss
import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer

faiss_index_path = "erp_data_faiss_chronosai_2.index"
metadata_path = "erp_metadata_2.json"

try:
    index = faiss.read_index(faiss_index_path)
except Exception:
    index = None

try:
    with open(metadata_path, "r", encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
except (FileNotFoundError, json.JSONDecodeError):
    metadata = []

try:
    embedding_model = SentenceTransformer("BAAI/bge-m3")
except Exception:
    exit()

def split_sentences(text):
    return re.split(r'(?<=[.!؟])\s+', text.strip()) if text else []

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def extract_best_sentence(query, chunk):
    sentences = split_sentences(chunk)
    if not sentences:
        return chunk  

    try:
        sentence_embeddings = embedding_model.encode(sentences, convert_to_numpy=True)
        query_embedding = embedding_model.encode(query, convert_to_numpy=True)

        similarities = np.dot(sentence_embeddings, query_embedding) / (
            np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        best_sentence_idx = np.argmax(softmax(similarities))
        return sentences[best_sentence_idx] if sentences else chunk
    except Exception:
        return chunk  

def retrieve_relevant_chunks(query, top_k=5):
    if not metadata or index is None:
        return ["⚠️ لا توجد بيانات متاحة."]

    try:
        query_embedding = embedding_model.encode(query, convert_to_numpy=True)
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        faiss.normalize_L2(query_embedding)
        _, indices = index.search(query_embedding, top_k)

        retrieved_sentences = [
            extract_best_sentence(query, metadata[i].get("content", "").strip())
            for i in indices[0] if 0 <= i < len(metadata)
        ]
        return [s for s in retrieved_sentences if len(s.split()) > 3] or ["❌ لا توجد معلومات."]
    except Exception:
        return ["⚠️ خطأ في البحث."]

if __name__ == "__main__":
    print(retrieve_relevant_chunks("ميزان المراجعة؟"))
