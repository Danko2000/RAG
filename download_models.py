from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# === 1. Download embedding model BGE ===
print("Downloading BGE model...")
bge = SentenceTransformer("BAAI/bge-base-en-v1.5")
bge.save("bge-base-en-v1.5")
print("Saved: bge-base-en-v1.5")

# === 2. Download Qwen2.5-3B-Instruct ===
print("Downloading Qwen tokenizer...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
tok.save_pretrained("qwen2.5-3b-instruct")

print("Downloading Qwen model (this may take several minutes)...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model.save_pretrained("qwen2.5-3b-instruct")

print("Done.")
