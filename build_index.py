# build_index.py
import os, time, json
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# === Конфиг ===
INPUT_DIR = r"C:\rasa\raw_sw"          # <- .md файлы с переименованными сущностями
INDEX_PATH = r"C:\rasa\sw_faiss.index"
METADATA_PATH = r"C:\rasa\sw_metadata.json"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 64  # при нехватке памяти: 16/32

print(f"[embed] loading model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)
EMB_DIM = model.get_sentence_embedding_dimension()
print(f"[embed] dim = {EMB_DIM}")

def load_markdown_files(root: str) -> List[Dict]:
    docs = []
    for fn in os.listdir(root):
        if fn.lower().endswith(".md"):
            path = os.path.join(root, fn)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append({"path": path, "title": os.path.splitext(fn)[0], "text": text})
    return docs

def chunk_documents(docs: List[Dict]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks, gid = [], 0
    for d in docs:
        raw = d["text"]
        parts = splitter.split_text(raw)
        offset = 0
        for i, t in enumerate(parts):
            start = raw.find(t, offset)
            if start == -1: start = offset
            end = start + len(t)
            offset = end
            chunks.append({
                "id": gid,
                "doc_title": d["title"],
                "doc_path": d["path"],
                "chunk_index": i,
                "start_char": start,
                "end_char": end,
                "text": t
            })
            gid += 1
    return chunks

def embed_passages(texts: List[str]) -> np.ndarray:
    # bge рекомендует префикс 'passage: '
    pref = [f"passage: {t}" for t in texts]
    embs = model.encode(
        pref, batch_size=BATCH_SIZE, convert_to_numpy=True,
        show_progress_bar=True, normalize_embeddings=True
    )
    return embs.astype("float32")

def build_faiss(embs: np.ndarray) -> faiss.IndexFlatIP:
    # embeddings уже нормализованы → Inner Product ~ cosine
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index

def main():
    t0 = time.time()
    print(f"[load] dir: {INPUT_DIR}")
    docs = load_markdown_files(INPUT_DIR)
    print(f"[load] files: {len(docs)}")

    print("[chunk] splitting ...")
    chunks = chunk_documents(docs)
    print(f"[chunk] total chunks: {len(chunks)}")

    print("[embed] encoding passages ...")
    texts = [c["text"] for c in chunks]
    embs = embed_passages(texts)
    print(f"[embed] shape: {embs.shape}")

    print("[faiss] building index ...")
    index = build_faiss(embs)

    print(f"[save] index -> {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)

    meta = [{
        "id": c["id"],
        "doc_title": c["doc_title"],
        "doc_path": c["doc_path"],
        "chunk_index": c["chunk_index"],
        "start_char": c["start_char"],
        "end_char": c["end_char"]
    } for c in chunks]

    print(f"[save] metadata -> {METADATA_PATH}")
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    dt = time.time() - t0
    print(f"[done] chunks={len(chunks)} time={dt:.1f}s")

if __name__ == "__main__":
    main()
