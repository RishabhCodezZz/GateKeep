"""NVIDIA NIM chat client with a disk cache that remembers each call's original latency."""
import hashlib, json, os, sqlite3, threading, time

from openai import OpenAI

NIM_URL = "https://integrate.api.nvidia.com/v1"
GEMMA = "google/gemma-4-31b-it"
ULTRA = "nvidia/nemotron-3-ultra-550b-a55b"
MIN_GAP = 60 / 40  # free tier: 40 requests/min shared across all models


class LLM:
    def __init__(self, cache_path="cache/llm.sqlite", client=None):
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        self.db = sqlite3.connect(cache_path, check_same_thread=False)
        self.db.execute("create table if not exists c (k text primary key, v text)")
        self.client = client or OpenAI(api_key=os.environ["NVIDIA_API_KEY"], base_url=NIM_URL)
        self.logical_calls = 0  # every chat() call, cached or not
        self.total_s = 0.0      # sum of ORIGINAL latencies, so cached reruns still report real speed
        self._last = 0.0
        self._lock = threading.Lock()

    def chat(self, model, prompt, system=None, max_tokens=512, temperature=0.0, extra=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        key = hashlib.sha256(json.dumps([model, msgs, max_tokens, temperature, extra]).encode()).hexdigest()
        self.logical_calls += 1
        row = self.db.execute("select v from c where k=?", (key,)).fetchone()
        if row:
            hit = json.loads(row[0])
            self.total_s += hit["s"]
            return hit["t"]
        with self._lock:
            wait = MIN_GAP - (time.time() - self._last)
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
