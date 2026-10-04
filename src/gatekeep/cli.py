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


@command
def questions(n_train="600", n_dev="100", n_test="300"):
    """Generate data/qa_{train,dev,test}.jsonl from chapter-disjoint chunks, plus data/review.csv for a hand check."""
    from gatekeep import qa
    from gatekeep.llm import LLM
    llm = LLM()
    chunks = [c for c in load_chunks() if c["chapter"] not in ("front", "back")]
    sp = qa.assign_splits([c["chapter"] for c in chunks])
    json.dump({k: sorted(v) for k, v in sp.items()}, open("data/splits.json", "w"), indent=1)
    topics = {"train": qa.ABSENT_TOPICS[:6], "dev": qa.ABSENT_TOPICS[6:8], "test": qa.ABSENT_TOPICS[8:]}
    for split, n in (("train", int(n_train)), ("dev", int(n_dev)), ("test", int(n_test))):
        part = qa.chunks_in(chunks, sp[split])
        items = qa.gen_single(llm, part, n) + qa.gen_multi(llm, part, n // 10)             + qa.gen_unanswerable(llm, chunks, topics[split])
        qa.save(qa.with_ids(items, split), f"data/qa_{split}.jsonl")
        print(split, len(items), "items,", sum(not it["answerable"] for it in items), "unanswerable,",
              len(part), "chunks in", len(sp[split]), "chapters")
    qa.review_sample(qa.load("data/qa_test.jsonl"), 50, "data/review.csv")


@command
def purge_empty():
    """Delete empty replies from the call cache (they were cached before the fix, so retries kept returning nothing)."""
    from gatekeep.llm import LLM
    print("purged", LLM().purge_empty(), "empty cached replies")


# --- new commands go above this line ---

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=sorted(COMMANDS))
    p.add_argument("args", nargs="*")
    a = p.parse_args()
    COMMANDS[a.cmd](*a.args)
