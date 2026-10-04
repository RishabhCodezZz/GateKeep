"""Build the question set. Generated from chunks of ONE split at a time, so chapters never leak."""
import csv, json, random, re

from gatekeep.llm import SUPER

ABSENT_TOPICS = ["LoRA", "retrieval-augmented generation", "vector database", "RLHF", "LangChain",
                 "diffusion model", "prompt engineering", "ChatGPT", "Hugging Face",
                 "tool calling", "LLM agents", "mixture of experts"]


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def assign_splits(chapters, seed=0, test_frac=0.25, dev_frac=0.10):
    cs = sorted(set(chapters))
    random.Random(seed).shuffle(cs)
    n_test, n_dev = max(1, round(len(cs) * test_frac)), max(1, round(len(cs) * dev_frac))
    return {"test": set(cs[:n_test]), "dev": set(cs[n_test:n_test + n_dev]), "train": set(cs[n_test + n_dev:])}


def chunks_in(chunks, chapters):
    return [c for c in chunks if c["chapter"] in chapters]


def _item(q, a, gold, chapter, answerable=True):
    return {"q": q, "a": a, "gold": gold, "chapter": chapter, "answerable": answerable, "src": "synthetic"}


def gen_single(llm, chunks, n, seed=0, model=SUPER):
    out = []
    for c in random.Random(seed).sample(chunks, min(n, len(chunks))):
        j = parse_json(llm.chat(model, "Write ONE question that can be answered only from the passage below, "
                                'plus a short answer. Reply with JSON only: {"question": "...", "answer": "..."}\n\n'
                                "Passage:\n" + c["text"], max_tokens=300))
        if j and j.get("question") and j.get("answer"):
            out.append(_item(j["question"], j["answer"], [c["id"]], c["chapter"]))
    return out


def gen_multi(llm, chunks, n, seed=0, model=SUPER):
    rng, out, tries = random.Random(seed), [], 0
    while len(out) < n and tries < 5 * n + 5 and len(chunks) >= 2:
        tries += 1
        a, b = rng.sample(chunks, 2)
        if a["chapter"] == b["chapter"]:
            continue
        j = parse_json(llm.chat(model, "Write ONE question that needs BOTH passages to answer, plus a short answer. "
                                'Reply with JSON only: {"question": "...", "answer": "..."}\n\n'
                                f"Passage 1:\n{a['text']}\n\nPassage 2:\n{b['text']}", max_tokens=300))
        if j and j.get("question") and j.get("answer"):
            out.append(_item(j["question"], j["answer"], [a["id"], b["id"]], a["chapter"]))
    return out


def gen_unanswerable(llm, all_chunks, topics, per_topic=8, model=SUPER):
    blob = " ".join(c["text"] for c in all_chunks).lower()
    out = []
    for t in topics:
        if re.search(r"\b" + re.escape(t.lower()) + r"\b", blob):  # a whole-word mention: not a valid trap
            continue
        j = parse_json(llm.chat(model, f'Write {per_topic} different questions a student might ask about "{t}". '
                                'JSON only: {"questions": ["..."]}', max_tokens=500))
        for q in (j or {}).get("questions", [])[:per_topic]:
            out.append(_item(q, None, [], "none", answerable=False))
    return out


def with_ids(items, prefix):
    return [{**it, "id": f"{prefix}-{i}"} for i, it in enumerate(items)]


def save(items, path):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(it) + "\n" for it in items)


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def review_sample(items, n, path, seed=0):
    rows = random.Random(seed).sample(items, min(n, len(items)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "answer", "ok(y/n)"])
        w.writerows([[r["id"], r["q"], r["a"], ""] for r in rows])
