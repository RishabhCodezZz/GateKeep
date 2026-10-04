"""Day-1 checks. Run on Kaggle (internet on, GPU on). Prints PASS/FAIL per check.

Backend comes from GATEKEEP_BACKEND (default "ollama"; needs OLLAMA_API_KEY). "nim" needs NVIDIA_API_KEY.
"""
import json, os, sys, time, urllib.request

sys.path.insert(0, "src")
from gatekeep.llm import GEMMA, ULTRA, LLM


def check(name, fn):
    try:
        print("PASS", name, "->", fn())
    except Exception as e:
        print("FAIL", name, "->", repr(e))


llm = LLM("cache/smoke.sqlite")
print("backend:", llm.backend)
check("gemma", lambda: repr(llm.chat(GEMMA, "Reply with the single word: ok", max_tokens=16)))
check("ultra", lambda: repr(llm.chat(ULTRA, "Reply with the single word: ok", max_tokens=16)))


def usage():
    # ponytail: /api/usage is reported by third parties, not confirmed in Ollama's docs; may 404 or change shape
    req = urllib.request.Request("https://ollama.com/api/usage",
                                 headers={"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"]})
    return json.load(urllib.request.urlopen(req, timeout=20))


def quota_probe(n=20):
    """How much free quota do n distinct short Nemotron calls use? Extrapolate to the ~6,000 Ultra calls we need."""
    before = usage()
    t0 = time.time()
    for i in range(n):
        llm.chat(ULTRA, f"Reply with the number {i}.", max_tokens=8)
    return f"{n} calls in {time.time() - t0:.0f}s\nbefore: {before}\nafter:  {usage()}"


if llm.backend == "ollama":
    check("ollama quota probe (Ultra)", quota_probe)


def laya_check():
    import laya
    agent = laya.load("convaiinnovations/laya", subfolder="typed-decisions")
    q = {"grade": {"type": "noul", "instructions": "Does the passage answer the question?"}}
    s = [{"body": "Question: What is overfitting?\nPassage: Overfitting means a model fits training noise."}]
    agent.predict_batch(s, q)  # warm-up
    t0 = time.perf_counter()
    r = agent.predict_batch(s * 16, q, batch_size=16)
    ms = (time.perf_counter() - t0) / 16 * 1000
    return f"P(true)={r[0]['answers']['grade']['noul']:.2f}, {ms:.1f} ms/decision"


check("laya", laya_check)
