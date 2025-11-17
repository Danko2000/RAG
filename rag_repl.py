# rag_repl.py
# -*- coding: utf-8 -*-

from rag_sw import RAGConfig, RAGEngine


def main():
    cfg = RAGConfig(
        index_path=r"C:\rasa\sw_faiss.index",
        metadata_path=r"C:\rasa\sw_metadata.json",
        embedding_model="BAAI/bge-base-en-v1.5",
        llm_model_name="Qwen/Qwen2.5-3B-Instruct",  # можно сменить, если есть ресурсы
        top_k=5,
        score_threshold=0.35,
        max_context_chars=3000,
        max_new_tokens=512,
        temperature=0.2,
        device="auto",   # "auto"|"cuda"|"cpu"
    )

    engine = RAGEngine(cfg)
    print("Локальный RAG (BGE + FAISS + Qwen2.5). Пиши вопрос, 'exit' или 'quit' для выхода.\n")

    while True:
        try:
            query = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if query.lower() in {"exit", "quit"}:
            print("Выход.")
            break

        if not query:
            continue

        result = engine.answer(query)

        print("\n=== ANSWER ===")
        print(result["answer"])

        print("\n=== SOURCES ===")
        if not result["retrieved"]:
            print("  (нет найденных чанков)")
        else:
            for i, item in enumerate(result["retrieved"], start=1):
                m = item["meta"]
                print(
                    f"[{i}] {m['doc_title']} "
                    f"(score={item['score']:.3f}, file={m['doc_path']})"
                )

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
