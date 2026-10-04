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
        items = (qa.gen_single(llm, part, n, llm_check=True) + qa.gen_multi(llm, part, n // 10, llm_check=True)
                 + qa.gen_unanswerable(llm, chunks, topics[split]))
        qa.save(qa.with_ids(items, split), f"data/qa_{split}.jsonl")
        print(split, len(items), "items,", sum(not it["answerable"] for it in items), "unanswerable,",
              len(part), "chunks in", len(sp[split]), "chapters")
    qa.review_sample(qa.load("data/qa_test.jsonl"), 50, "data/review.csv")


@command
def purge_empty():
    """Delete empty replies from the call cache (they were cached before the fix, so retries kept returning nothing)."""
    from gatekeep.llm import LLM
    print("purged", LLM().purge_empty(), "empty cached replies")


def load_system():
    from gatekeep import qa
    from gatekeep.corpus import Index
    from gatekeep.llm import LLM
    return LLM(), Index(load_chunks()), qa


def read_rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


@command
def variants(names="V0,V1", limit="", tau_file="results/tau.json", agents="0", prefix="", items_file="data/qa_test.jsonl"):
    """Run end-to-end variants; write results/<prefix>rows_<variant>.jsonl and append to results/summary.csv.
    prefix keeps side runs apart (e.g. "tau0.9_", "ood_", "hw_"); `report` only reads rows_*.jsonl."""
    import csv
    from gatekeep import evalrun
    llm, index, qa = load_system()
    items = qa.load(items_file)
    items = items[:int(limit)] if limit else items
    laya = None
    if any(v in names for v in ("V2", "V3")):
        from gatekeep.gates import LayaGate
        laya = LayaGate.load(json.load(open("models/ckpts.json")))
    tau = json.load(open(tau_file)) if Path(tau_file).exists() else 0.8
    Path("results").mkdir(exist_ok=True)
    summaries = {}
    for v in names.split(","):
        app, llm_gate = evalrun.make_app(v, index, llm, laya, tau, agents=agents == "1")
        rows = evalrun.run_variant(app, llm_gate, llm, items)
        suffix = "-agents" if agents == "1" else ""
        qa.save(rows, f"results/{prefix}rows_{v}{suffix}.jsonl")
        tag = prefix + v + suffix
        summaries[tag] = evalrun.summarize(rows)
        print(tag, {k: round(m[0], 3) for k, m in summaries[tag].items()}, "| empty replies:", llm.empty)
    with open("results/summary.csv", "a", newline="") as f:
        w = csv.writer(f)
        for tag, s in summaries.items():
            for k, (m, lo, hi) in s.items():
                w.writerow([tag, k, round(m, 4), round(lo, 4), round(hi, 4)])


@command
def gatedata():
    """Write data/gates/<gate>_<split>.jsonl (+ .laya.jsonl) for train/dev/test."""
    from gatekeep import data
    llm, index, qa = load_system()
    by_id = {c["id"]: c for c in index.chunks}
    direct = data.gen_direct(llm, 40)
    cut = {"train": direct[:28], "dev": direct[28:32], "test": direct[32:]}
    for split in ("train", "dev", "test"):
        items = qa.load(f"data/qa_{split}.jsonl")
        real = split != "train"
        data.write_gate_data({"route": data.g1_rows(items, cut[split]),
                              "grade": data.g2_rows(llm, index, items, by_id),
                              "grounded": data.g3_rows(llm, index, items, by_id, real=real),
                              "sufficient": data.g4_rows(llm, items)}, split)
        print(split, "done")


@command
def handwritten(path="handwritten.csv"):
    """Convert your question,answer CSV into data/handwritten.jsonl (a blank answer = the book cannot answer it)."""
    from gatekeep import qa
    items = qa.load_handwritten_csv(path)
    Path("data").mkdir(exist_ok=True)
    qa.save(items, "data/handwritten.jsonl")
    print(len(items), "questions,", sum(not it["answerable"] for it in items), "unanswerable")


@command
def check_filter(review="review.csv"):
    """Score the question filters against the y/n marks in review.csv (answerable rows only)."""
    import csv
    from gatekeep import qa
    from gatekeep.llm import LLM
    llm = LLM()
    rows = [r for r in list(csv.reader(open(review, encoding="utf-8-sig")))[1:] if r[2].strip() and r[3].strip()]
    table = Counter()
    for _id, q, _a, mark in rows:
        pattern_ok = not qa.bad_question(q)
        both_ok = pattern_ok and qa.self_contained(llm, q)
        table[("patterns", mark.lower(), "kept" if pattern_ok else "rejected")] += 1
        table[("patterns+judge", mark.lower(), "kept" if both_ok else "rejected")] += 1
    n_bad = sum(1 for r in rows if r[3].lower() == "n")
    for stage in ("patterns", "patterns+judge"):
        print(f"{stage:15s} bad questions removed: {table[(stage, 'n', 'rejected')]}/{n_bad} | "
              f"good questions lost: {table[(stage, 'y', 'rejected')]}/{len(rows) - n_bad}")


# --- new commands go above this line ---

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=sorted(COMMANDS))
    p.add_argument("args", nargs="*")
    a = p.parse_args()
    COMMANDS[a.cmd](*a.args)
