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
    """Parse the PDF with Docling and write data/book.docling.json + data/chunks.json."""
    from gatekeep import corpus
    Path("data").mkdir(exist_ok=True)
    doc = corpus.parse(pdf)
    doc.save_as_json(Path("data/book.docling.json"))
    chunks = corpus.chunk(doc)
    json.dump(chunks, open("data/chunks.json", "w"))
    print(len(chunks), "chunks")
    for ch, n in Counter(c["chapter"] for c in chunks).most_common():
        print(f"{n:5d}  {ch}")


# --- new commands go above this line ---

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=sorted(COMMANDS))
    p.add_argument("args", nargs="*")
    a = p.parse_args()
    COMMANDS[a.cmd](*a.args)
