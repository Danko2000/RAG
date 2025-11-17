import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
RAW_DIR = PROJECT_ROOT / "raw_sw"
OUT_DIR = PROJECT_ROOT / "knowledge_base"
TERMS_MAP_PATH = PROJECT_ROOT / "terms_map.json"

def load_terms_map():
    with open(TERMS_MAP_PATH, "r", encoding="utf-8") as f:
        terms = json.load(f)
    # сортируем по длине ключей, чтобы сначала заменять длинные фразы
    # (например "The Force" до "Force")
    sorted_terms = sorted(terms.items(), key=lambda kv: len(kv[0]), reverse=True)
    return sorted_terms

def replace_terms_in_text(text: str, sorted_terms):
    for src, dst in sorted_terms:
        text = text.replace(src, dst)
    return text

def generate_output_filename(src_path: Path, sorted_terms) -> str:
    """Пытаемся подобрать осмысленное имя файла по самому важному термину.
       Если не нашли — оставляем старое имя."""
    name = src_path.stem
    new_name = name
    for src, dst in sorted_terms:
        if src.lower().replace(" ", "_") in name.lower():
            # например luke_skywalker -> rial_solari
            new_name = dst.lower().replace(" ", "_")
            break
    return new_name + ".md"

def main():
    sorted_terms = load_terms_map()
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    for file_path in RAW_DIR.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        replaced = replace_terms_in_text(text, sorted_terms)
        out_name = generate_output_filename(file_path, sorted_terms)
        out_path = OUT_DIR / out_name

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(replaced)

        print(f"Processed {file_path.name} -> {out_name}")

if __name__ == "__main__":
    main()
