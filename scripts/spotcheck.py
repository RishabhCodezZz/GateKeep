"""Compare Docling's chapters with the PDF's own bookmarks and print random chunks. Prints BOOK TEXT: keep the notebook private."""
import json, random, sys

import pymupdf

pdf = sys.argv[1]
chunks = json.load(open("data/chunks.json"))
toc = [t[1] for t in pymupdf.open(pdf).get_toc(simple=True) if t[0] == 1]
print("BOOKMARKS (level 1):", *toc, sep="\n  ")
print("\nCHAPTERS FROM DOCLING:", *sorted({c["chapter"] for c in chunks}), sep="\n  ")
longest = max(len(c["ctx"]) for c in chunks)
print(f"\n{len(chunks)} chunks; longest contextualized chunk = {longest} chars")
for c in random.Random(0).sample(chunks, 8):
    print("\n---", c["chapter"], "|", c["section"], "\n", c["text"][:400])
