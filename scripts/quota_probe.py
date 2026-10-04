"""Measure Ollama quota per realistic call. Needs only a CPU Kaggle session (no GPU).

Sends N distinct chunk-sized prompts to each model, shaped like our real calls, and reports how much of the
monthly quota they used. The meter rounds to 3 decimals, so N must be large (default 100 per model).
"""
import sys, time
sys.path.insert(0, "src")
from gatekeep.llm import GEMMA, SUPER, ULTRA, LLM, ollama_usage

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
ONLY = sys.argv[2] if len(sys.argv) > 2 else ""  # e.g. "super" runs only probes whose name contains it
llm = LLM("cache/quota_probe.sqlite", backend="ollama")
PASSAGE = ("Overfitting happens when a model learns the noise in the training set instead of the underlying "
           "pattern. Regularization, more data, simpler models and early stopping all reduce it. ") * 4  # ~300 tokens


def probe(name, model, make_prompt, max_tokens):
    if ONLY not in name:
        return
    before, t0 = ollama_usage(), time.time()
    for i in range(N):
        llm.chat(model, make_prompt(i), max_tokens=max_tokens)
    after = ollama_usage()
    per_call = (after - before) / N
    cap = f"{1 / per_call:,.0f}" if per_call > 0 else "more than " + f"{N / 0.001:,.0f}"
    print(f"{name}: {N} calls in {time.time() - t0:.0f}s, usage {before} -> {after}, "
          f"~{per_call:.6f} per call, ~{cap} calls = 100% of the month")


# Gemma shaped like a gate / writer call: ~350-token prompt, short or medium answer
probe("gemma gate-style (max 16 tokens)", GEMMA,
      lambda i: f"Question {i}: why does a model overfit?\nPassage: {PASSAGE}\nDoes the passage help answer the question? Reply YES or NO.", 16)
probe("gemma writer-style (max 200 tokens)", GEMMA,
      lambda i: f"Answer using only the passage.\n\n{PASSAGE}\n\nQuestion {i}: how can overfitting be reduced?\nAnswer:", 200)
# Ultra shaped like question generation / labeling
probe("ultra labeler-style (max 150 tokens)", ULTRA,
      lambda i: f"Write ONE question about passage {i} plus a short answer. JSON only.\n\nPassage:\n{PASSAGE}", 150)
# Same shape on Super (120B, 12B active): the likely cheaper stand-in for Ultra
probe("super labeler-style (max 150 tokens)", SUPER,
      lambda i: f"Write ONE question about passage {i} plus a short answer. JSON only.\n\nPassage:\n{PASSAGE}", 150)
