"""Training/eval rows for the four gates. Labels come from the book's own structure or from Super."""
import json
from pathlib import Path

from gatekeep.gates import GATES
from gatekeep.graph import evidence, gen_prompt
from gatekeep.llm import GEMMA, SUPER
from gatekeep.qa import parse_json


def _yes(text):
    return text.strip().upper().startswith("YES")


def gen_direct(llm, n=40):
    j = parse_json(llm.chat(SUPER, f"Write {n} short messages that need no document lookup: greetings, thanks, "
                                   "'what is 2+2'-style trivia. JSON only: {\"messages\": [\"...\"]}", max_tokens=1500))
    return (j or {}).get("messages", [])


def g1_rows(items, direct_texts):
    rows = [{"text": it["q"], "label": "retrieve" if it["answerable"] else "out_of_scope"} for it in items]
    return rows + [{"text": t, "label": "direct"} for t in direct_texts]


def g2_rows(llm, index, items, by_id, negs=2):
    rows = []
    for it in items:
        if not it["answerable"]:
            continue
        gold = set(it["gold"])
        rows += [{"text": f"Question: {it['q']}\nPassage: {by_id[c]['text']}", "label": "yes"} for c in it["gold"]]
        for h in [h for h in index.search(it["q"], 20) if h["id"] not in gold][:negs]:
            ans = llm.chat(SUPER, f"Question: {it['q']}\nPassage: {h['text']}\nDoes the passage contain information "
                                  "needed to answer the question? Reply YES or NO.", max_tokens=8)
            rows.append({"text": f"Question: {it['q']}\nPassage: {h['text']}", "label": "yes" if _yes(ans) else "no"})
    return rows


def g3_rows(llm, index, items, by_id, real=False):
    """real=False: faithful answer vs one-fact-corrupted answer (training). real=True: Gemma's answer from
    retrieved chunks, labeled by Super (dev/test)."""
    rows = []
    for it in items:
        if not it["answerable"]:
            continue
        docs = index.rerank(it["q"], index.search(it["q"], 20), 5) if real else [by_id[c] for c in it["gold"]]
        good = llm.chat(GEMMA, gen_prompt(it["q"], docs, ""), max_tokens=300)
        ev = f"Passages:\n{evidence(docs)}\n\nAnswer: "
        if real:
            v = llm.chat(SUPER, f"{ev}{good}\nIs every claim in the answer supported by the passages? Reply YES or NO.",
                         max_tokens=8)
            rows.append({"text": ev + good, "label": "yes" if _yes(v) else "no"})
        else:
            bad = llm.chat(SUPER, "Rewrite this answer so that exactly one factual claim is wrong (change a number, "
                                  "term or relation). Keep the style. Reply with the answer only.\n\n" + good, max_tokens=300)
            rows += [{"text": ev + good, "label": "yes"}, {"text": ev + bad, "label": "no"}]
    return rows


def g4_rows(llm, items):
    rows = []
    for it in items:
        if not it["answerable"]:
            continue
        evasive = llm.chat(SUPER, "Write a plausible answer that discusses the same topic but does NOT actually "
                                  f"answer this question. Reply with the answer only.\n\nQuestion: {it['q']}", max_tokens=200)
        rows += [{"text": f"Question: {it['q']}\nAnswer: {it['a']}", "label": "yes"},
                 {"text": f"Question: {it['q']}\nAnswer: {evasive}", "label": "no"}]
    return rows


def laya_row(gate, text, label):
    # ponytail: the README does not show how noul answers are encoded; confirm against the official notebook (plan Task 8 step 6)
    ans = {"choice": label} if GATES[gate]["q"]["type"] == "choice" else {"noul": label == "yes"}
    return {"state": {"body": text}, "questions": {gate: GATES[gate]["q"]}, "answers": {gate: ans}}


def write_gate_data(rows_by_gate, split, outdir="data/gates"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for gate, rows in rows_by_gate.items():
        with open(f"{outdir}/{gate}_{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        with open(f"{outdir}/{gate}_{split}.laya.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(laya_row(gate, r["text"], r["label"])) + "\n" for r in rows)
