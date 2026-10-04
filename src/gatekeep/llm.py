"""Chat client for Gemma/Nemotron (Ollama Cloud or NVIDIA NIM) with a disk cache that remembers each call's original latency."""
import hashlib, json, os, sqlite3, threading, time, urllib.request

from openai import OpenAI

GEMMA, ULTRA, SUPER = "gemma", "ultra", "super"  # logical names, resolved to a model id per backend
BACKENDS = {
    # Ollama Cloud free tier: unpublished GPU-time quota, 1 concurrent request (our client is serial)
    "ollama": dict(url="https://ollama.com/v1", key="OLLAMA_API_KEY", gap=0.0,
                   models={GEMMA: "gemma4:31b", ULTRA: "nemotron-3-ultra", SUPER: "nemotron-3-super"}),
    # NVIDIA NIM free tier: 40 requests/min shared across models
    "nim": dict(url="https://integrate.api.nvidia.com/v1", key="NVIDIA_API_KEY", gap=60 / 40,
                models={GEMMA: "google/gemma-4-31b-it", ULTRA: "nvidia/nemotron-3-ultra-550b-a55b",
                        SUPER: "nvidia/nemotron-3-super-120b-a12b"}),
}


class LLM:
    def __init__(self, cache_path="cache/llm.sqlite", client=None, backend=None):
        self.backend = backend or os.environ.get("GATEKEEP_BACKEND", "ollama")
        self.cfg = BACKENDS[self.backend]
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        self.db = sqlite3.connect(cache_path, check_same_thread=False)
        self.db.execute("create table if not exists c (k text primary key, v text)")
        self.client = client or OpenAI(api_key=os.environ[self.cfg["key"]], base_url=self.cfg["url"])
        self.logical_calls = 0  # every chat() call, cached or not
        self.total_s = 0.0      # sum of ORIGINAL latencies, so cached reruns still report real speed
        self._last = 0.0
        self._lock = threading.Lock()

    def chat(self, model, prompt, system=None, max_tokens=512, temperature=0.0, extra=None):
        model = self.cfg["models"].get(model, model)
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        key = hashlib.sha256(json.dumps([self.backend, model, msgs, max_tokens, temperature, extra]).encode()).hexdigest()
        self.logical_calls += 1
        row = self.db.execute("select v from c where k=?", (key,)).fetchone()
        if row:
            hit = json.loads(row[0])
            self.total_s += hit["s"]
            return hit["t"]
        with self._lock:
            wait = self.cfg["gap"] - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            t0 = time.perf_counter()
            r = self.client.chat.completions.create(
                model=model, messages=msgs, max_tokens=max_tokens,
                temperature=temperature, extra_body=extra)
            s = time.perf_counter() - t0
            self._last = time.time()
        text = r.choices[0].message.content or ""
        self.total_s += s
        self.db.execute("insert or replace into c values (?,?)", (key, json.dumps({"t": text, "s": s})))
        self.db.commit()
        return text


def ollama_usage():
    """Fraction (0-1, rounded to 3 decimals) of the monthly Ollama Cloud quota used so far.
    ponytail: /api/usage is not in Ollama's docs; seen working on 2026-10-04 (3-decimal rounding)."""
    req = urllib.request.Request("https://ollama.com/api/usage",
                                 headers={"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"]})
    return json.load(urllib.request.urlopen(req, timeout=20))["limits"]["monthly"]["usage"]
