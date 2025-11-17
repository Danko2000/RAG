# build_index.py
# -*- coding: utf-8 -*-
"""
Создаёт FAISS-индекс по Markdown-документам и пишет подробную статистику.

- Читает .md из INPUT_DIR (по умолчанию: C:\rasa\raw_sw или /app/raw_sw в Docker)
- Делит тексты на чанки (RecursiveCharacterTextSplitter)
- Кодирует эмбеддинги локальной моделью BAAI/bge-base-en-v1.5
- Строит FAISS IndexFlatIP (inner product) с L2-нормализованными эмбеддингами
- Сохраняет:
    - INDEX_PATH    (по умолчанию: C:\rasa\sw_faiss.index или /app/data/sw_faiss.index)
    - METADATA_PATH (по умолчанию: C:\rasa\sw_metadata.json или /app/data/sw_metadata.json)
    - STATS_PATH    (по умолчанию: C:\rasa\sw_stats.json или /app/data/sw_stats.json)
"""

import os
import json
import argparse
from time import perf_counter
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

try:
    # Новый пакет-расщепитель
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # На случай установленного "старого" langchain
    from langchain.text_splitter import RecursiveCharacterTextSplitter


# ----------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ----------------------------

def _default_path(env_name: str, win_default: str, linux_default: str) -> str:
    """
    Берём путь из ENV, если задан, иначе — платформенный дефолт.
    """
    env_val = os.getenv(env_name)
    if env_val:
        return env_val
    return win_default if os.name == "nt" else linux_default


def load_markdown_files(root: str) -> List[Dict]:
    docs: List[Dict] = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Input dir not found: {root}")

    for fn in os.listdir(root):
        if not fn.lower().endswith(".md"):
            continue
        path = os.path.join(root, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            # fallback для экзотических кодировок
            with open(path, "r", encoding="cp1251", errors="replace") as f:
                text = f.read()
        docs.append({"path": path, "title": os.path.splitext(fn)[0], "text": text})
    return docs


def chunk_documents(
    docs: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Dict]:
    """
    Делим документы на логические чанки через RecursiveCharacterTextSplitter.
    В каждом чанке сохраняем:
    - id (глобальный id чанка)
    - doc_title, doc_path
    - chunk_index (номер чанка внутри документа)
    - start_char, end_char (позиция чанка в исходном тексте)
    - text (сам текст чанка)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: List[Dict] = []
    gid = 0

    for d in docs:
        raw = d["text"]
        parts = splitter.split_text(raw)
        offset = 0
        for i, t in enumerate(parts):
            # Пробуем восстановить позицию чанка в исходном тексте
            start = raw.find(t, offset)
            if start == -1:
                start = offset
            end = start + len(t)
            offset = end
            chunks.append({
                "id": gid,
                "doc_title": d["title"],
                "doc_path": d["path"],
                "chunk_index": i,
                "start_char": start,
                "end_char": end,
                "text": t,
            })
            gid += 1

    return chunks


def embed_passages(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Кодируем тексты в эмбеддинги BGE.
    Рекомендация BGE: префиксовать документы "passage: ".
    """
    prefixed = [f"passage: {t}" for t in texts]
    embs = model.encode(
        prefixed,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 → inner product ~= cosine
    )
    return embs.astype("float32")


def build_faiss(embs: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index


def sizeof_mb(path: str) -> float:
    try:
        return round(os.path.getsize(path) / (1024 * 1024), 3)
    except Exception:
        return 0.0


# ----------------------------
# MAIN
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index over Markdown docs with BGE embeddings."
    )

    # Пути по умолчанию: ENV → платформа
    default_input_dir = _default_path("RAG_RAW_DIR", r"C:\rasa\raw_sw", "/app/raw_sw")
    default_index_path = _default_path("RAG_INDEX_PATH", r"C:\rasa\sw_faiss.index", "/app/data/sw_faiss.index")
    default_meta_path = _default_path("RAG_METADATA_PATH", r"C:\rasa\sw_metadata.json", "/app/data/sw_metadata.json")
    default_stats_path = _default_path("RAG_STATS_PATH", r"C:\rasa\sw_stats.json", "/app/data/sw_stats.json")

    # Модель: по умолчанию имя на HF, но можно передать локальный путь через ENV/CLI
    default_model = os.getenv("RAG_EMBEDDING_MODEL_PATH", "BAAI/bge-base-en-v1.5")

    parser.add_argument("--input_dir", default=default_input_dir, help="Папка с .md файлами")
    parser.add_argument("--index_path", default=default_index_path, help="Куда сохранить индекс")
    parser.add_argument("--meta_path", default=default_meta_path, help="Куда сохранить метаданные чанков")
    parser.add_argument("--stats_path", default=default_stats_path, help="Куда сохранить статистику")
    parser.add_argument("--model", default=default_model, help="Модель эмбеддингов (имя HF или локальный путь)")
    parser.add_argument("--device", default=None, help="cpu|cuda (опционально, если torch с CUDA установлен)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch для encode()")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Размер чанка (символы)")
    parser.add_argument("--chunk_overlap", type=int, default=150, help="Перекрытие чанков (символы)")

    args = parser.parse_args()

    print(f"[cfg] input_dir      = {args.input_dir}")
    print(f"[cfg] index_path     = {args.index_path}")
    print(f"[cfg] meta_path      = {args.meta_path}")
    print(f"[cfg] stats_path     = {args.stats_path}")
    print(f"[cfg] model          = {args.model}")
    print(f"[cfg] device         = {args.device or '(auto)'}")
    print(f"[cfg] batch_size     = {args.batch_size}")
    print(f"[cfg] chunk_size     = {args.chunk_size}")
    print(f"[cfg] chunk_overlap  = {args.chunk_overlap}")

    t_total_0 = perf_counter()

    # --- Load embedding model ---
    t0 = perf_counter()
    model_kwargs = {}
    if args.device:
        model_kwargs["device"] = args.device  # 'cpu'|'cuda'
    # Внутри Docker ожидаем локальный путь → работаем offline
    model = SentenceTransformer(args.model, local_files_only=True, **model_kwargs)
    emb_dim = model.get_sentence_embedding_dimension()
    t1 = perf_counter()
    time_model_load = t1 - t0
    print(f"[embed] model loaded: {args.model} (dim={emb_dim}, dt={time_model_load:.3f}s)")

    # --- Load files ---
    t0 = perf_counter()
    docs = load_markdown_files(args.input_dir)
    t1 = perf_counter()
    time_load_files = t1 - t0
    print(f"[load] files={len(docs)} dt={time_load_files:.3f}s")

    if not docs:
        raise RuntimeError(f"No .md files found in {args.input_dir}")

    # --- Chunking ---
    t0 = perf_counter()
    chunks = chunk_documents(
        docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    t1 = perf_counter()
    time_chunking = t1 - t0
    print(f"[chunk] chunks={len(chunks)} dt={time_chunking:.3f}s")

    # Доп. агрегаты для статистики
    total_chars = sum(len(c["text"]) for c in chunks) if chunks else 0
    total_words = sum(len(c["text"].split()) for c in chunks) if chunks else 0
    avg_chars = round(total_chars / max(len(chunks), 1), 1)
    avg_words = round(total_words / max(len(chunks), 1), 1)

    # --- Embeddings ---
    t0 = perf_counter()
    texts = [c["text"] for c in chunks]
    embs = embed_passages(texts, model, batch_size=args.batch_size)
    t1 = perf_counter()
    time_embeddings = t1 - t0
    print(f"[embed] shape={embs.shape} dt={time_embeddings:.3f}s")

    # --- FAISS build ---
    t0 = perf_counter()
    index = build_faiss(embs)
    t1 = perf_counter()
    time_faiss_build = t1 - t0
    print(f"[faiss] index.ntotal={index.ntotal} dt={time_faiss_build:.3f}s")

    if index.ntotal != len(chunks):
        print("[warn] index.ntotal != chunks_count (рассинхрон).")

    # ensure dirs
    os.makedirs(os.path.dirname(args.index_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.meta_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.stats_path), exist_ok=True)

    # --- Save index ---
    t0 = perf_counter()
    faiss.write_index(index, args.index_path)
    t1 = perf_counter()
    time_save_index = t1 - t0
    index_mb = sizeof_mb(args.index_path)
    print(f"[save] index -> {args.index_path} (dt={time_save_index:.3f}s, size={index_mb} MB)")

    # --- Save metadata ---
    t0 = perf_counter()
    meta = [{
        "id": c["id"],
        "doc_title": c["doc_title"],
        "doc_path": c["doc_path"],
        "chunk_index": c["chunk_index"],
        "start_char": c["start_char"],
        "end_char": c["end_char"],
    } for c in chunks]
    with open(args.meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    t1 = perf_counter()
    time_save_meta = t1 - t0
    meta_mb = sizeof_mb(args.meta_path)
    print(f"[save] metadata -> {args.meta_path} (dt={time_save_meta:.3f}s, size={meta_mb} MB)")

    # --- Stats ---
    total_time = perf_counter() - t_total_0

    stats = {
        "model_name": args.model,
        "embedding_dim": int(emb_dim),
        "input_dir": args.input_dir,
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "avg_chunk_chars": avg_chars,
        "avg_chunk_words": avg_words,
        "index_path": args.index_path,
        "metadata_path": args.meta_path,
        "index_vectors": int(index.ntotal),
        "index_file_mb": index_mb,
        "metadata_file_mb": meta_mb,
        "times_sec": {
            "model_load": round(time_model_load, 3),
            "load_files": round(time_load_files, 3),
            "chunking": round(time_chunking, 3),
            "embeddings": round(time_embeddings, 3),
            "faiss_build": round(time_faiss_build, 3),
            "save_index": round(time_save_index, 3),
            "save_meta": round(time_save_meta, 3),
            "total": round(total_time, 3),
        },
        "throughput": {
            "chunks_per_sec_embed": round(len(chunks) / time_embeddings, 2) if time_embeddings > 0 else None,
        },
    }

    with open(args.stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Человекочитаемый вывод
    print("\n===== STATS =====")
    print(f"Docs:              {stats['n_docs']}")
    print(f"Chunks:            {stats['n_chunks']}")
    print(f"Index vectors:     {stats['index_vectors']}")
    print(f"Index size (MB):   {stats['index_file_mb']}")
    print(f"Metadata size (MB):{stats['metadata_file_mb']}")
    print(f"Avg chunk:         {stats['avg_chunk_chars']} chars / {stats['avg_chunk_words']} words")
    print("Times (sec):")
    for k, v in stats["times_sec"].items():
        print(f"  {k:12s}: {v}")
    print(f"Saved stats -> {args.stats_path}")


if __name__ == "__main__":
    main()
