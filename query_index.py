# query_index.py
import os, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = r"C:\rasa\sw_faiss.index"
METADATA_PATH = r"C:\rasa\sw_metadata.json"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

print(f"[embed] loading model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)

def load_index_and_meta():
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

def embed_query(q: str) -> np.ndarray:
    vec = model.encode([f"query: {q}"], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vec.astype("float32").reshape(1, -1)

def main():
    index, meta = load_index_and_meta()
    query = input("Enter query: ").strip()
    qv = embed_query(query)
    scores, idxs = index.search(qv, 5)

    print("\nTop results:")
    for s, i in zip(scores[0], idxs[0]):
        m = meta[int(i)]
        print("-" * 80)
        print(f"score: {float(s):.3f}")
        print(f"doc_title:   {m['doc_title']}")
        print(f"doc_path:    {m['doc_path']}")
        print(f"chunk_index: {m['chunk_index']}")
        print(f"position:    {m['start_char']}–{m['end_char']}")
        with open(m["doc_path"], "r", encoding="utf-8") as f:
            full = f.read()
        snippet = full[m["start_char"]:m["end_char"]]
        print("\nsnippet:")
        print(snippet[:500], "..." if len(snippet) > 500 else "")

if __name__ == "__main__":
    main()
