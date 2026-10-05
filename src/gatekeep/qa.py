"""Build the question set. Generated from chunks of ONE split at a time, so chapters never leak."""
import csv, json, random, re

from gatekeep.llm import SUPER

ABSENT_TOPICS = ["LoRA", "retrieval-augmented generation", "vector database", "RLHF", "LangChain",
                 "diffusion model", "prompt engineering", "ChatGPT", "Hugging Face",
                 "tool calling", "LLM agents", "mixture of experts"]

# Written after a hand check of 50 generated questions: 31 were unusable. The patterns below catch 25 of those 31
# and wrongly reject 1 of 19 good ones; the LLM check (self_contained) handles the rest.
RULES = ("The question must make sense to a student who has NOT seen the passage: never say 'the passage', 'the text', "
         "'the example' or 'the code', and do not ask about a specific number, array, variable or printed output. "
         "Ask about a concept, a method, a reason or a definition. Give a short answer of one or two sentences.")
PASSAGE_REF = re.compile(r"\b(passages?|the text|excerpts?|shown (above|below)|as shown|this example|"
                         r"the example (above|below|project)|in example \d|figure \d|table \d)\b", re.I)
NUMBER_PROBE = re.compile(r"\b(value of|the (median|mean|average|probability|reward|accuracy|rmse|score) (of|for|value)|"
                          r"the (first|second|third|fourth|fifth) (element|instance|feature|row|item)|in state \S+|"
                          r"action \d|first instance)\b", re.I)
CODE_VARIABLE = re.compile(r"\b\w+_\w*\s+(array|dataset|variable|attribute|list|matrix|table)\b", re.I)
# Found in a second hand check (47 answerable questions, 16 still bad); 5 of those 16 matched, 1 of 31 good ones did.
CODE_REF = re.compile(r"\b(code snippet|snippet|defined in|described|provided (information|passage)|given (information|passage)|"
                      r"the following|in the code|the code|the function defined|function defined|\w+_\w+\(\)\s+function)\b", re.I)
SYMBOLS = set("()[]{}=_<>/*#\\")
CODE_LIMIT = 0.03  # share of symbol characters; bad questions came from chunks at a median 0.026-0.033 vs 0.007-0.011 for good ones


def code_heavy(text, limit=CODE_LIMIT):
    return sum(ch in SYMBOLS for ch in text) / max(len(text), 1) >= limit


def bad_question(q):
    return bool(PASSAGE_REF.search(q) or NUMBER_PROBE.search(q) or CODE_VARIABLE.search(q) or CODE_REF.search(q))


def self_contained(llm, q, model=SUPER):
    r = llm.chat(model, "Could a student who has not seen this textbook's specific examples, code listings, figures or "
                        "datasets answer this question from general machine-learning knowledge or the textbook's "
                        f"explanations? Reply YES or NO.\n\nQuestion: {q}", max_tokens=8)
    return r.strip().upper().startswith("YES")


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


def _usable(llm, j, model, llm_check):
    if not (j and j.get("question") and j.get("answer")):
        return False
    return not bad_question(j["question"]) and (not llm_check or self_contained(llm, j["question"], model))


def gen_single(llm, chunks, n, seed=0, model=SUPER, llm_check=False):
    """Walk the prose chunks in random order until n usable questions exist (or the chunks run out)."""
    order = [c for c in chunks if not code_heavy(c["text"])]
    random.Random(seed).shuffle(order)
    out = []
    for c in order:
        if len(out) >= n:
            break
        j = parse_json(llm.chat(model, "Write ONE question that can be answered from the passage below. " + RULES +
                                ' Reply with JSON only: {"question": "...", "answer": "..."}\n\n'
                                "Passage:\n" + c["text"], max_tokens=300))
        if _usable(llm, j, model, llm_check):
            out.append(_item(j["question"], j["answer"], [c["id"]], c["chapter"]))
    return out


def gen_multi(llm, chunks, n, seed=0, model=SUPER, llm_check=False):
    chunks = [c for c in chunks if not code_heavy(c["text"])]
    rng, out, tries = random.Random(seed), [], 0
    while len(out) < n and tries < 5 * n + 5 and len(chunks) >= 2:
        tries += 1
        a, b = rng.sample(chunks, 2)
        if a["chapter"] == b["chapter"]:
            continue
        j = parse_json(llm.chat(model, "Write ONE question that needs BOTH passages to answer. " + RULES +
                                ' Reply with JSON only: {"question": "...", "answer": "..."}\n\n'
                                f"Passage 1:\n{a['text']}\n\nPassage 2:\n{b['text']}", max_tokens=300))
        if _usable(llm, j, model, llm_check):
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


def load_handwritten_csv(path):
    """question,answer rows -> items; a blank answer means the book cannot answer it."""
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    return [{"id": f"hw-{i}", "q": r["question"].strip(), "a": r["answer"].strip() or None, "gold": [],
             "chapter": "hand", "answerable": bool(r["answer"].strip()), "src": "hand"}
            for i, r in enumerate(rows) if r["question"].strip()]


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
