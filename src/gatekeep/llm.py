"""Chat client for Gemma/Nemotron (Ollama Cloud or NVIDIA NIM) with a disk cache that remembers each call's original latency."""
import hashlib, json, os, sqlite3, threading, time, urllib.request

from openai import OpenAI

GEMMA, ULTRA, SUPER = "gemma", "ultra", "super"  # logical names, resolved to a model id per backend
BACKENDS = {
    # Ollama Cloud free tier: unpublished GPU-time quota, 1 concurrent request (our client is serial)
    "ollama": dict(url="https://ollama.com/v1", key="OLLAMA_API_KEY", gap=0.0,
                   models={GEMMA: "gemma4:31b", ULTRA: "nemotron-3-ultra", SUPER: "nemotron-3-super"},
                   # thinking off: with it on, long prompts spent the whole token budget thinking and returned "" (measured
                   # 2026-10-04: Super 144 -> 38 tokens). Gemma already defaults to off and "low" would turn it ON, so send nothing.
                   extra={SUPER: {"reasoning_effort": "none"}, ULTRA: {"reasoning_effort": "none"}}),
    # NVIDIA NIM free tier: 40 requests/min shared across models
    "nim": dict(url="https://integrate.api.nvidia.com/v1", key="NVIDIA_API_KEY", gap=60 / 40,
                models={GEMMA: "google/gemma-4-31b-it", ULTRA: "nvidia/nemotron-3-ultra-550b-a55b",
                        SUPER: "nvidia/nemotron-3-super-120b-a12b"},
                extra={}),  # ponytail: NIM's thinking switch is untested; find it before routing a Nemotron model here
}


class LLM:
    """routes = {logical_model: backend} sends one model to another provider (env GATEKEEP_ROUTES="super=nim,ultra=ollama")."""

    def __init__(self, cache_path="cache/llm.sqlite", client=None, backend=None, routes=None):
        self.backend = backend or os.environ.get("GATEKEEP_BACKEND", "ollama")
        BACKENDS[self.backend]  # a typo fails here, not on the first call
        if routes is None:
            routes = dict(p.strip().split("=") for p in os.environ.get("GATEKEEP_ROUTES", "").split(",") if p.strip())
        self.routes = {m: b.strip() for m, b in routes.items()}
        for b in self.routes.values():
            BACKENDS[b]
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        self.db = sqlite3.connect(cache_path, check_same_thread=False)
        self.db.execute("create table if not exists c (k text primary key, v text)")
        self._fake, self._clients = client, {}  # tests pass one fake client that serves every backend
        self.logical_calls = 0  # every chat() call, cached or not
        self.empty = 0          # real calls that came back empty
        self.last_tokens = None  # completion tokens of the latest call (replayed from the cache on a hit)
        self.total_s = 0.0      # sum of ORIGINAL latencies, so cached reruns still report real speed
        self._last = {}         # per backend: time of the last real call
        self._lock = threading.Lock()

    def _client(self, backend):
        if self._fake:
            return self._fake
        if backend not in self._clients:
            cfg = BACKENDS[backend]
            self._clients[backend] = OpenAI(api_key=os.environ[cfg["key"]], base_url=cfg["url"])
        return self._clients[backend]

    def chat(self, model, prompt, system=None, max_tokens=512, temperature=0.0, extra=None):
        backend = self.routes.get(model, self.backend)
        cfg = BACKENDS[backend]
        if extra is None:
            extra = cfg["extra"].get(model)
        model = cfg["models"].get(model, model)
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        key = hashlib.sha256(json.dumps([backend, model, msgs, max_tokens, temperature, extra]).encode()).hexdigest()
        self.logical_calls += 1
        row = self.db.execute("select v from c where k=?", (key,)).fetchone()
        if row:
            hit = json.loads(row[0])
            self.total_s += hit["s"]
            self.last_tokens = hit.get("n")
            return hit["t"]
        with self._lock:
            wait = cfg["gap"] - (time.time() - self._last.get(backend, 0.0))
            if wait > 0:
                time.sleep(wait)
            t0 = time.perf_counter()
            r = self._client(backend).chat.completions.create(
                model=model, messages=msgs, max_tokens=max_tokens,
                temperature=temperature, extra_body=extra)
            s = time.perf_counter() - t0
            self._last[backend] = time.time()
        text = r.choices[0].message.content or ""
        self.last_tokens = getattr(getattr(r, "usage", None), "completion_tokens", None)
        self.total_s += s
        if not text.strip():  # a reasoning model that spent its whole budget thinking: never cache nothing
            self.empty += 1
            return text
        self.db.execute("insert or replace into c values (?,?)", (key, json.dumps({"t": text, "s": s, "n": self.last_tokens})))
        self.db.commit()
        return text

    def purge_empty(self):
        """Delete cached empty replies (written before empty replies stopped being cached); returns how many."""
        dead = [k for k, v in self.db.execute("select k, v from c") if not json.loads(v)["t"].strip()]
        self.db.executemany("delete from c where k=?", [(k,) for k in dead])
        self.db.commit()
        return len(dead)


def ollama_usage():
    """Fraction (0-1, rounded to 3 decimals) of the monthly Ollama Cloud quota used so far.
    ponytail: /api/usage is not in Ollama's docs; seen working on 2026-10-04 (3-decimal rounding)."""
    req = urllib.request.Request("https://ollama.com/api/usage",
                                 headers={"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"]})
    return json.load(urllib.request.urlopen(req, timeout=20))["limits"]["monthly"]["usage"]
