"""Command line entry: python -m gatekeep.cli <command> [args...]"""
import argparse, json
from collections import Counter
from pathlib import Path

COMMANDS = {}


def _usage():
    try:
        from gatekeep.llm import ollama_usage
        return ollama_usage()
    except Exception:  # no key, other backend, or the endpoint changed
        return None


def command(f):
    def run(*a):
        u0 = _usage()
        try:
            return f(*a)
        finally:
            print(f"[ollama monthly usage {u0} -> {_usage()}]")
    COMMANDS[f.__name__] = run
    return f


def load_chunks(path="data/chunks.json"):
    return json.load(open(path))


@command
def prepare(pdf):
    """Parse the PDF (reusing data/book.docling.json if present), chunk it by PDF-bookmark chapters, write data/chunks.json."""
    from gatekeep import corpus
    Path("data").mkdir(exist_ok=True)
    saved = Path("data/book.docling.json")
    if saved.exists():
        doc = corpus.load(saved)
    else:
        doc = corpus.parse(pdf)
        doc.save_as_json(saved)
    starts = corpus.chapter_starts(pdf)
    chunks = corpus.chunk(doc, starts=starts)
    json.dump(chunks, open("data/chunks.json", "w"))
    counts = Counter(c["chapter"] for c in chunks)
    print(len(chunks), "chunks")
    for ch in ["front"] + [t for _, t in starts]:
        print(f"{counts.get(ch, 0):5d}  {ch}")


# --- new commands go above this line ---

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=sorted(COMMANDS))
    p.add_argument("args", nargs="*")
    a = p.parse_args()
    COMMANDS[a.cmd](*a.args)
