"""Check chapters (from PDF bookmarks) and chunk sizes (in tokens), then print random chunks.
The last part prints BOOK TEXT: keep the notebook private."""
import json, random, sys
from collections import Counter

sys.path.insert(0, "src")
from gatekeep.corpus import EMB, chapter_starts

pdf = sys.argv[1]
chunks = json.load(open("data/chunks.json"))
starts = chapter_starts(pdf)
counts = Counter(c["chapter"] for c in chunks)

print(f"{len(chunks)} chunks; expected chapters 1-19 + 'back': {len(starts)} bookmark entries")
print("CHUNKS PER CHAPTER (book order):")
for ch in ["front"] + [t for _, t in starts]:
    print(f"  {counts.get(ch, 0):5d}  {ch}")
missing = [t for _, t in starts if not counts.get(t)]
print("chapters with NO chunks:", missing or "none")
print("chunks without page info:", sum(c["page"] is None for c in chunks))

from transformers import AutoTokenizer  # bge tokenizer: close to, not the same as, Laya's ModernBERT tokenizer
tok = AutoTokenizer.from_pretrained(EMB)
lens = sorted(len(tok(c["text"], add_special_tokens=False)["input_ids"]) for c in chunks)
print(f"\nTOKENS per chunk text: median {lens[len(lens) // 2]}, p95 {lens[int(len(lens) * .95)]}, max {lens[-1]} "
      f"(Laya window is ~512 and also holds the question; over 450: {sum(n > 450 for n in lens)} chunks)")

for c in random.Random(0).sample(chunks, 6):
    print("\n---", c["chapter"], "| p.", c["page"], "|", c["section"], "\n", c["text"][:400])
