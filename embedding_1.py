import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-m3")

file_path = "cleaned_final_erp_data.jsonl"

metadata = []
with open(file_path, "r", encoding="utf-8") as meta_file:
    for line in meta_file:
        try:
            metadata.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            continue

documents = list({item.get("content", "").strip() for item in metadata if item.get("content", "").strip()})

if documents:
    embeddings = embedding_model.encode(documents, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss_index_path = "erp_data_faiss_chronos_2.index"
    faiss.write_index(index, faiss_index_path)

    meta_path = "erp_metadata_2.json"
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump([{"content": doc} for doc in documents], meta_file, ensure_ascii=False, indent=4)
else:
    print("No valid data found for indexing.")
