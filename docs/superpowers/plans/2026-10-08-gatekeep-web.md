# GateKeep Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local website (`http://127.0.0.1:8000`) where the author asks GateKeep questions from the book or the scikit-learn docs, in Fast (V2) or Careful (V3) mode, and watches a live gate-by-gate trace next to each answer.

**Architecture:** `demo.ask_events` runs one question through the existing LangGraph pipeline and yields events. `web.py` is a standard-library HTTP server that serves three static files and streams those events as JSON lines. A `web` CLI command loads the models and indexes once and starts the server. The page is plain HTML, CSS and JS.

**Tech Stack:** Python 3.12 (`.venv`), `http.server`, LangGraph pipeline from `gatekeep.graph`, Laya gates, Ollama Cloud, plain HTML/CSS/JS, Google Fonts (Kalam, Inter, JetBrains Mono).

**Spec:** `docs/superpowers/specs/2026-10-08-gatekeep-web-design.md`

## Global Constraints

- Always use the project venv: `.\.venv\Scripts\python.exe` (never the global Python). CLI calls need `$env:PYTHONPATH = "src"`, and anything that calls Ollama needs `$env:OLLAMA_API_KEY = [Environment]::GetEnvironmentVariable('OLLAMA_API_KEY','User')` in the same PowerShell command (never print the key).
- No new Python packages: the server uses the standard library only.
- The server binds to `127.0.0.1` only.
- Fonts: Kalam for the app name and the welcome heading only; Inter for questions, answers and the trace; JetBrains Mono for timings and code.
- Colours: background `#FAF9F6`, text `#252525`, primary `#5268B3`.
- Never commit book-derived data: `data/`, `cache/`, `models/`, `*.pdf`, `*rows*.jsonl`, zips.
- Commit messages have no attribution lines. Do not push unless a task says so.
- Prose (README) follows the humanizer rules: no em or en dashes.

## File map

- Modify `src/gatekeep/demo.py`: add `who()`, `_gates()`, `ask_events()`; rebuild `ask()` and `gate_line()` on them.
- Create `src/gatekeep/static/index.html`, `src/gatekeep/static/app.css`, `src/gatekeep/static/app.js`.
- Create `src/gatekeep/web.py`: `make_server(corpora, ask, port=8000)`.
- Modify `src/gatekeep/corpus.py`: `Index(chunks, device=None)`.
- Modify `src/gatekeep/cli.py`: `prepare_dir(folder, out="data/chunks.json")`, `WEB_CORPORA`, `web(port="8000")`.
- Create `tests/test_web.py`; modify `tests/test_demo.py`.
- Modify `README.md` (a "Web app" section).

---

### Task 1: `ask_events`, a streaming version of the demo run

**Files:**
- Modify: `src/gatekeep/demo.py`
- Test: `tests/test_demo.py`

**Interfaces:**
- Consumes: `gatekeep.graph.build_graph(index, llm, gates)`, `gatekeep.graph.REFUSAL`, `gatekeep.gates.GATES`, `Cascade(fast, slow, tau)`, `LLMGate(llm)`; the existing `Recorder` and `describe` in `demo.py`.
- Produces: `ask_events(question, index, llm, laya, taus, mode="careful")`, a generator of dicts:
  - `{"type": "step", "node": str, "text": str}`
  - `{"type": "gate", "gate": str, "labels": list[str], "n": int, "who": str, "ms": int}` where `who` is `"Laya"` or `"Laya, then Gemma for k of n"`
  - `{"type": "passages", "items": [{"text": str, "section": str, "page": int | None}]}`
  - `{"type": "answer", "text": str, "refused": bool, "reason": "off_topic" | "no_passage" | None}`
  - `{"type": "done", "seconds": float}`
  - `mode` is `"fast"` (every gate by Laya) or `"careful"` (Cascade per gate); anything else raises `ValueError`.
  - `ask()` keeps its signature and return value `(answer, trace_lines, passage_texts, seconds)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_demo.py`:
```python
from gatekeep.demo import ask_events


class CountingLLM(FakeLLM):
    def __init__(self):
        super().__init__()
        self.gate_prompts = 0

    def chat(self, model, prompt, **kw):
        if prompt.startswith(("Does this", "Does the", "Is every", "Is this")):
            self.gate_prompts += 1  # the LLM gate's questions
        return super().chat(model, prompt, **kw)


def test_events_come_in_order_and_end_with_answer_then_done():
    evs = list(ask_events("what is bagging?", FakeIndex(), FakeLLM(), laya(1.0), TAUS, "careful"))
    kinds = [e["type"] for e in evs]
    assert kinds[0] == "step" and evs[0]["node"] == "route"
    assert kinds[-3:] == ["passages", "answer", "done"]
    assert evs[-2] == {"type": "answer", "text": "an answer", "refused": False, "reason": None}
    assert len(evs[-3]["items"]) == 5 and set(evs[-3]["items"][0]) == {"text", "section", "page"}
    gates = [e for e in evs if e["type"] == "gate"]
    assert {g["gate"] for g in gates} == {"route", "grade", "grounded", "sufficient"} and all(g["who"] == "Laya" for g in gates)


def test_fast_mode_never_asks_the_llm_to_judge_even_when_laya_is_unsure():
    llm = CountingLLM()
    evs = list(ask_events("what is bagging?", FakeIndex(), llm, laya(0.1), TAUS, "fast"))
    assert llm.gate_prompts == 0 and all(e["who"] == "Laya" for e in evs if e["type"] == "gate")


def test_careful_mode_hands_unsure_decisions_to_gemma():
    evs = list(ask_events("what is bagging?", FakeIndex(), FakeLLM(), laya(0.1), TAUS, "careful"))
    assert any(e["type"] == "gate" and e["who"] == "Laya, then Gemma for 1 of 1" for e in evs)


def test_refusals_carry_their_reason():
    off = list(ask_events("capital of France?", FakeIndex(), FakeLLM(), laya(1.0, "off_topic"), TAUS, "fast"))
    assert off[-2] == {"type": "answer", "text": REFUSAL, "refused": True, "reason": "off_topic"}
    nothing = Scripted(lambda g, t: ("retrieve", 1.0) if g == "route" else ("no", 1.0))
    none = list(ask_events("x", FakeIndex(), FakeLLM(), nothing, TAUS, "fast"))
    assert none[-2]["reason"] == "no_passage" and none[-3]["items"] == []


def test_unknown_mode_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        list(ask_events("x", FakeIndex(), FakeLLM(), laya(1.0), TAUS, "turbo"))
```

- [ ] **Step 2: Run them to see them fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_demo.py -q`
Expected: `ImportError: cannot import name 'ask_events'`.

- [ ] **Step 3: Implement**

In `src/gatekeep/demo.py`, replace `gate_line` and `ask` with the code below and add `who`, `_gates` and `ask_events` (keep `Recorder`, `describe` and `launch` as they are):
```python
def who(e):
    return "Laya" if not e["escalated"] else f"Laya, then Gemma for {e['escalated']} of {e['n']}"


def gate_line(e):
    labels = e["labels"][0] if e["n"] == 1 else f"{e['labels'].count('yes')} of {e['n']} yes"
    return f"   {e['gate']}: {labels} ({e['who']}, {e['ms']} ms)"


def _gates(mode, laya, llm, taus, log):
    if mode == "fast":   # V2: every gate by Laya
        return {g: Recorder(laya, log) for g in GATES}
    if mode == "careful":  # V3: Laya first, Gemma when Laya is unsure
        return {g: Recorder(Cascade(laya, LLMGate(llm), taus[g]), log) for g in GATES}
    raise ValueError(f"unknown mode {mode!r}: use 'fast' or 'careful'")


def ask_events(question, index, llm, laya, taus, mode="careful"):
    """Run one question and yield what happens, step by step, for the trace and the web app."""
    log, t0 = [], time.perf_counter()
    app = build_graph(index, llm, _gates(mode, laya, llm, taus, log))
    docs, answer, route = [], "", None
    for update in app.stream({"q": question}, stream_mode="updates"):
        for node, out in update.items():
            yield {"type": "step", "node": node, "text": describe(node, out)}
            for e in log:
                yield {"type": "gate", "gate": e["gate"], "labels": e["labels"], "n": e["n"], "who": who(e), "ms": e["ms"]}
            log.clear()
            docs, answer, route = out.get("docs", docs), out.get("answer", answer), out.get("route", route)
    refused = answer == REFUSAL
    yield {"type": "passages", "items": [{"text": d["text"], "section": d.get("section", ""), "page": d.get("page")}
                                         for d in ([] if refused else docs)]}
    yield {"type": "answer", "text": answer, "refused": refused,
           "reason": ("off_topic" if route == "off_topic" else "no_passage") if refused else None}
    yield {"type": "done", "seconds": round(time.perf_counter() - t0, 2)}


def ask(question, index, llm, laya, taus, source="the scikit-learn user guide"):
    """Run one question through the V3 cascade. Returns (answer, trace lines, passages, seconds)."""
    trace, passages, answer, secs = [], [], "", 0.0
    for ev in ask_events(question, index, llm, laya, taus, "careful"):
        if ev["type"] == "step":
            trace.append(ev["text"])
        elif ev["type"] == "gate":
            trace.append(gate_line(ev))
        elif ev["type"] == "passages":
            passages = [p["text"] for p in ev["items"]]
        elif ev["type"] == "answer":  # the pipeline's fixed refusal says "this book"; name the corpus the app uses
            answer = f"I can't answer that from {source}." if ev["refused"] else ev["text"]
        elif ev["type"] == "done":
            secs = ev["seconds"]
    return answer, trace, passages, secs
```

- [ ] **Step 4: Run the tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_demo.py -q` → all pass (the three existing `ask` tests too). Then the full suite once: `.\.venv\Scripts\python.exe -m pytest -q`.

- [ ] **Step 5: Commit**

```powershell
git add src/gatekeep/demo.py tests/test_demo.py
git commit -m "feat: ask_events streams each step of a question; ask() is built on it"
```

---

### Task 2: The page (HTML, CSS, JS)

**Files:**
- Create: `src/gatekeep/static/index.html`, `src/gatekeep/static/app.css`, `src/gatekeep/static/app.js`

**Interfaces:**
- Consumes (from Task 3, defined here so this task can be written first): `GET /api/info` → `{"corpora": [{"name", "label", "ready", "prepare", "examples"}]}`; `POST /api/ask` with JSON `{"question", "mode", "corpus"}` → a stream of JSON lines (the Task 1 events, plus `{"type": "error", "message"}`); a 400 reply is JSON `{"error": str}`.
- Produces: the three static files served by Task 3 at `/` and `/static/app.css`, `/static/app.js`.

This task has no unit tests; Task 5 checks it in a browser.

- [ ] **Step 1: `src/gatekeep/static/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GateKeep</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono&family=Kalam:wght@400;700&display=swap">
<link rel="stylesheet" href="/static/app.css">
</head>
<body>
<aside class="sidebar">
  <h1 class="brand">GateKeep</h1>
  <fieldset class="switch" id="corpus"><legend>Answer from</legend></fieldset>
  <fieldset class="switch" id="mode"><legend>Mode</legend>
    <label><input type="radio" name="mode" value="fast" checked> Fast (Laya)</label>
    <label><input type="radio" name="mode" value="careful"> Careful (Laya + Gemma)</label>
  </fieldset>
  <button type="button" id="new-chat" class="secondary">New chat</button>
  <p class="note">Each question is answered on its own.</p>
  <p class="note small">Careful makes about 8 Gemma calls per answer, Fast about 1.5.</p>
</aside>
<main>
  <section id="welcome" class="welcome">
    <h2>What would you like to know?</h2>
    <svg class="underline" viewBox="0 0 300 14" aria-hidden="true"><path d="M4 9 C 70 3, 150 13, 296 6"/></svg>
    <div id="examples" class="examples"></div>
  </section>
  <ol id="chat" class="chat" aria-live="polite"></ol>
  <form id="ask" class="composer">
    <textarea id="question" rows="2" placeholder="Ask about machine learning" aria-label="Your question"></textarea>
    <button type="submit" id="send">Send</button>
  </form>
</main>
<script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: `src/gatekeep/static/app.css`**

```css
:root {
  --bg: #FAF9F6; --ink: #252525; --primary: #5268B3; --muted: #6b6b6b;
  --line: #e4e1da; --panel: #ffffff; --soft: #eef0f8; --err: #b3261e;
  --sans: "Inter", system-ui, sans-serif; --hand: "Kalam", cursive; --mono: "JetBrains Mono", ui-monospace, monospace;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.6 var(--sans);
       display: grid; grid-template-columns: 260px 1fr; min-height: 100vh; }
.sidebar { border-right: 1px solid var(--line); padding: 24px 20px; display: flex; flex-direction: column; gap: 18px; }
.brand { font: 700 30px/1 var(--hand); margin: 0; }
.switch { border: 0; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 6px; }
.switch legend { font-weight: 600; margin-bottom: 4px; }
.switch label { display: flex; gap: 8px; align-items: center; cursor: pointer; }
.switch label.off { color: var(--muted); cursor: not-allowed; }
.switch .hint { font-size: 12px; color: var(--muted); margin-left: 24px; }
button { font: inherit; border-radius: 10px; border: 1px solid var(--primary); background: var(--primary); color: #fff;
         padding: 8px 16px; cursor: pointer; }
button.secondary { background: transparent; color: var(--primary); }
button:disabled { opacity: .5; cursor: not-allowed; }
.note { margin: 0; color: var(--muted); font-size: 14px; }
.note.small { font-size: 12px; }
main { display: flex; flex-direction: column; max-width: 860px; width: 100%; margin: 0 auto; padding: 24px 24px 0; }
.welcome { text-align: center; margin-top: 12vh; }
.welcome h2 { font: 700 34px/1.2 var(--hand); margin: 0; }
.underline { width: 300px; max-width: 80%; height: 14px; }
.underline path { fill: none; stroke: var(--primary); stroke-width: 2.5; stroke-linecap: round; }
.examples { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin-top: 24px; }
.examples button { background: var(--panel); color: var(--ink); border-color: var(--line); }
.chat { list-style: none; padding: 0; margin: 0; flex: 1; display: flex; flex-direction: column; gap: 28px; }
.q { align-self: flex-end; background: var(--soft); border-radius: 14px; padding: 10px 14px; max-width: 80%; white-space: pre-wrap; }
.a { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px; font-size: 17px; }
.a p { margin: 0 0 10px; white-space: pre-wrap; }
.a p:last-child { margin-bottom: 0; }
.a.pending { color: var(--muted); }
.a.refused { color: var(--muted); background: transparent; border-style: dashed; }
.cite { padding: 0 4px; font-size: 13px; background: var(--soft); color: var(--primary); border: 0; border-radius: 6px; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.chips button { padding: 2px 10px; font-size: 13px; background: var(--panel); color: var(--primary); border-color: var(--line); }
.passage { margin-top: 8px; padding: 10px 12px; border-left: 3px solid var(--primary); background: var(--panel); font-size: 14px; white-space: pre-wrap; }
.passage .meta { display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }
.trace { margin-top: 10px; font-size: 14px; }
.trace summary { cursor: pointer; color: var(--muted); }
.trace ol { list-style: none; padding: 8px 0 0 4px; margin: 0; display: flex; flex-direction: column; gap: 4px; }
.trace .gate { padding-left: 14px; }
.badge { font-size: 12px; padding: 1px 8px; border-radius: 999px; margin-left: 6px; }
.badge.laya { background: var(--primary); color: #fff; }
.badge.gemma { background: #d9d9d9; color: var(--ink); }
.ms, .total { font-family: var(--mono); font-size: 12px; color: var(--muted); margin-left: 6px; }
.error { color: var(--err); margin-top: 8px; }
.composer { position: sticky; bottom: 0; background: var(--bg); display: flex; gap: 10px; padding: 16px 0 20px; }
.composer textarea { flex: 1; font: inherit; resize: vertical; border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px; background: var(--panel); }
@media (max-width: 760px) {
  body { grid-template-columns: 1fr; }
  .sidebar { border-right: 0; border-bottom: 1px solid var(--line); flex-direction: row; flex-wrap: wrap; align-items: center; padding: 12px 16px; gap: 12px; }
  .sidebar .note { display: none; }
  .welcome { margin-top: 6vh; }
  main { padding: 16px 16px 0; }
}
```

- [ ] **Step 3: `src/gatekeep/static/app.js`**

```js
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
let corpora = [];
let busy = false;

const picked = (name) => document.querySelector(`input[name=${name}]:checked`)?.value;
const corpusLabel = () => corpora.find((c) => c.name === picked("corpus"))?.label ?? "the corpus";

async function load() {
  corpora = (await (await fetch("/api/info")).json()).corpora;
  const box = $("#corpus");
  const first = corpora.find((c) => c.ready);
  for (const c of corpora) {
    const label = el("label", c.ready ? "" : "off");
    const input = Object.assign(document.createElement("input"), { type: "radio", name: "corpus", value: c.name, disabled: !c.ready, checked: c === first });
    label.append(input, " " + c.label);
    box.append(label);
    if (!c.ready) box.append(el("div", "hint", "Not prepared: " + c.prepare));
  }
  box.addEventListener("change", showExamples);
  showExamples();
}

function showExamples() {
  const box = $("#examples");
  box.replaceChildren();
  for (const q of corpora.find((c) => c.name === picked("corpus"))?.examples ?? []) {
    const b = el("button", "", q);
    b.type = "button";
    b.onclick = () => send(q);
    box.append(b);
  }
}

function renderAnswer(box, text, open) {
  box.replaceChildren();
  for (const para of text.split(/\n{2,}/)) {
    const p = el("p");
    for (const part of para.split(/(\[\d+\])/)) {
      const m = part.match(/^\[(\d+)\]$/);
      if (m) { const b = el("button", "cite", part); b.type = "button"; b.onclick = () => open(Number(m[1]) - 1); p.append(b); }
      else p.append(part);
    }
    box.append(p);
  }
}

function newTurn(question) {
  const li = el("li", "turn");
  const answer = el("div", "a pending", "Checking…");
  const chips = el("div", "chips");
  const trace = el("details", "trace");
  trace.open = true;
  trace.append(el("summary", "", "How this answer was checked"));
  const rows = el("ol");
  trace.append(rows);
  li.append(el("p", "q", question), answer, chips, trace);
  $("#chat").append(li);
  let items = [];
  const open = (i) => {
    li.querySelector(".passage")?.remove();
    const p = items[i];
    if (!p) return;
    const box = el("div", "passage");
    box.append(el("span", "meta", [p.section, p.page != null ? "page " + p.page : ""].filter(Boolean).join(" · ") || "passage " + (i + 1)), p.text);
    chips.after(box);
  };
  const scroll = () => li.scrollIntoView({ block: "end", behavior: "smooth" });
  return {
    event(ev) {
      if (ev.type === "step") rows.append(el("li", "step", ev.text));
      else if (ev.type === "gate") {
        const row = el("li", "gate", `${ev.gate}: ${ev.n === 1 ? ev.labels[0] : ev.labels.filter((l) => l === "yes").length + " of " + ev.n + " yes"}`);
        row.append(el("span", "badge " + (ev.who.includes("Gemma") ? "gemma" : "laya"), ev.who), el("span", "ms", ev.ms + " ms"));
        rows.append(row);
      } else if (ev.type === "passages") {
        items = ev.items;
        items.forEach((_, i) => { const b = el("button", "", `[${i + 1}]`); b.type = "button"; b.onclick = () => open(i); chips.append(b); });
      } else if (ev.type === "answer") {
        answer.className = "a" + (ev.refused ? " refused" : "");
        if (!ev.refused) renderAnswer(answer, ev.text, open);
        else answer.textContent = ev.reason === "off_topic" ? "Not about machine learning, so I didn't look it up."
          : `No passage in ${corpusLabel()} answered this (query rewritten twice).`;
      } else if (ev.type === "done") rows.append(el("li", "total", `total ${ev.seconds.toFixed(1)} s`));
      else if (ev.type === "error") this.error(ev.message);
      scroll();
    },
    error(msg) { answer.classList.remove("pending"); li.append(el("p", "error", "Something went wrong: " + msg)); scroll(); },
  };
}

async function send(question) {
  question = question.trim();
  if (!question || busy || !picked("corpus")) return;
  busy = true; $("#send").disabled = true; $("#welcome").hidden = true; $("#question").value = "";
  const turn = newTurn(question);
  try {
    const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode: picked("mode"), corpus: picked("corpus") }) });
    if (!res.ok) { turn.error((await res.json()).error); return; }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) { turn.event(JSON.parse(buf.slice(0, i))); buf = buf.slice(i + 1); }
    }
  } catch (e) {
    turn.error(String(e));
  } finally {
    busy = false; $("#send").disabled = false; $("#question").focus();
  }
}

$("#ask").addEventListener("submit", (e) => { e.preventDefault(); send($("#question").value); });
$("#question").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send($("#question").value); } });
$("#new-chat").addEventListener("click", () => { if (!busy) { $("#chat").replaceChildren(); $("#welcome").hidden = false; } });
load();
```

- [ ] **Step 4: Commit**

```powershell
git add src/gatekeep/static
git commit -m "feat: GateKeep web page (chat, live gate trace, source chips, corpus and mode switches)"
```

---

### Task 3: The local server

**Files:**
- Create: `src/gatekeep/web.py`
- Test: `tests/test_web.py`

**Interfaces:**
- Consumes: the static files from Task 2 (`src/gatekeep/static/`).
- Produces: `make_server(corpora, ask, port=8000) -> http.server.ThreadingHTTPServer` bound to `127.0.0.1` (port 0 picks a free port; the chosen one is `server.server_address[1]`).
  - `corpora`: `{name: {"label": str, "index": object | None, "prepare": str, "examples": list[str]}}`; `index is None` means not ready.
  - `ask(question, index, mode)`: an iterator of event dicts (Task 1's events). Exceptions it raises become one `{"type": "error", "message": ...}` line.

- [ ] **Step 1: Write the failing tests**

`tests/test_web.py`:
```python
import json
import threading
import urllib.error
import urllib.request

import pytest

from gatekeep.web import make_server


def fake_ask(question, index, mode):
    if question == "boom":
        raise RuntimeError("no network")
    yield {"type": "step", "node": "route", "text": f"route -> retrieve ({mode}, {index})"}
    yield {"type": "done", "seconds": 0.1}


CORPORA = {"book": {"label": "Book", "index": "book-index", "prepare": "", "examples": ["q1"]},
           "sklearn": {"label": "scikit-learn docs", "index": None, "prepare": "python scripts/fetch_sklearn_docs.py",
                       "examples": []}}


@pytest.fixture
def url():
    srv = make_server(CORPORA, fake_ask, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url, path):
    return urllib.request.urlopen(url + path, timeout=5)


def post(url, body):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(url + "/api/ask", data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=5)


def status(fn, *a):
    try:
        return fn(*a).status
    except urllib.error.HTTPError as e:
        return e.code


def test_serves_the_page_and_its_files(url):
    page = get(url, "/")
    assert page.headers["Content-Type"].startswith("text/html") and b"GateKeep" in page.read()
    assert get(url, "/static/app.js").headers["Content-Type"].startswith("text/javascript")
    assert status(get, url, "/static/../web.py") == 404 and status(get, url, "/static/nope.css") == 404


def test_info_lists_corpora_and_which_are_ready(url):
    info = json.load(get(url, "/api/info"))
    assert [(c["name"], c["ready"]) for c in info["corpora"]] == [("book", True), ("sklearn", False)]
    assert info["corpora"][1]["prepare"].startswith("python scripts/fetch")


def test_ask_streams_json_lines_ending_in_done(url):
    res = post(url, {"question": "what is bagging?", "mode": "fast", "corpus": "book"})
    assert res.headers["Content-Type"] == "application/x-ndjson"
    events = [json.loads(line) for line in res.read().decode().splitlines()]
    assert events[0]["text"] == "route -> retrieve (fast, book-index)" and events[-1]["type"] == "done"


@pytest.mark.parametrize("body", [{"question": "  ", "mode": "fast", "corpus": "book"},
                                  {"question": "q", "mode": "turbo", "corpus": "book"},
                                  {"question": "q", "mode": "fast", "corpus": "sklearn"},
                                  {"question": "q", "mode": "fast", "corpus": "nope"},
                                  {"mode": "fast"}, b"not json"])
def test_bad_requests_get_400(url, body):
    assert status(post, url, body) == 400


def test_a_failing_backend_becomes_an_error_event(url):
    events = [json.loads(line) for line in post(url, {"question": "boom", "mode": "careful", "corpus": "book"}).read().decode().splitlines()]
    assert events == [{"type": "error", "message": "RuntimeError: no network"}]
```

- [ ] **Step 2: Run them to see them fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_web.py -q`
Expected: `ModuleNotFoundError: No module named 'gatekeep.web'`.

- [ ] **Step 3: Implement `src/gatekeep/web.py`**

```python
"""Local web app: serves the page and streams one question's events as JSON lines (standard library only)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).parent / "static"
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8"}
MODES = ("fast", "careful")


def make_server(corpora, ask, port=8000):
    """corpora: {name: {"label", "index" (None = not ready), "prepare", "examples"}}; ask(question, index, mode) -> events."""
    lock = threading.Lock()  # one GPU: one question at a time

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the console quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _line(self, event):
            self.wfile.write((json.dumps(event) + "\n").encode())
            self.wfile.flush()

        def do_GET(self):
            if self.path == "/api/info":
                return self._send(200, {"corpora": [{"name": n, "label": c["label"], "ready": c["index"] is not None,
                                                     "prepare": c["prepare"], "examples": c["examples"]}
                                                    for n, c in corpora.items()]})
            name = "index.html" if self.path == "/" else self.path.removeprefix("/static/")
            f = STATIC / name
            if "/" in name or "\\" in name or f.suffix not in TYPES or not f.is_file():  # flat folder only
                return self._send(404, {"error": "not found"})
            self._send(200, f.read_bytes(), TYPES[f.suffix])

        def do_POST(self):
            if self.path != "/api/ask":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                question, mode, corpus = req["question"].strip(), req["mode"], req["corpus"]
            except (ValueError, KeyError, TypeError, AttributeError):
                return self._send(400, {"error": "send JSON with question, mode and corpus"})
            if not question or mode not in MODES or corpora.get(corpus, {}).get("index") is None:
                return self._send(400, {"error": "empty question, unknown mode, or a corpus that is not ready"})
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            with lock:
                try:
                    for event in ask(question, corpora[corpus]["index"], mode):
                        self._line(event)
                except Exception as e:  # shown on this turn; the server keeps running
                    self._line({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
```

- [ ] **Step 4: Run the tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_web.py -q` → 10 passed. Then the full suite once.

- [ ] **Step 5: Commit**

```powershell
git add src/gatekeep/web.py tests/test_web.py
git commit -m "feat: local standard-library server that streams a question's events as JSON lines"
```

---

### Task 4: The `web` command and a separate docs corpus

**Files:**
- Modify: `src/gatekeep/corpus.py:62-67` (`Index.__init__`)
- Modify: `src/gatekeep/cli.py` (`prepare_dir`, new `WEB_CORPORA`, new `web` command above the `# --- new commands go above this line ---` marker)

**Interfaces:**
- Consumes: `gatekeep.demo.ask_events` (Task 1), `gatekeep.web.make_server` (Task 3), `gatekeep.gates.LayaGate.load(ckpts)`, `gatekeep.llm.LLM`, `cli.load_chunks(path)`.
- Produces: `Index(chunks, device=None)`; `prepare_dir(folder, out="data/chunks.json")`; command `web(port="8000")`.

These are wiring changes over code that loads real models, so they are checked by running them in Task 5, not by unit tests.

- [ ] **Step 1: `Index` can stay on the CPU**

In `src/gatekeep/corpus.py`, change the start of `Index` to:
```python
class Index:
    def __init__(self, chunks, device=None):
        """device=None picks the GPU when there is one; the web app passes "cpu" to leave the 4 GB GPU to the Laya gates."""
        import faiss
        import numpy as np
        from sentence_transformers import CrossEncoder, SentenceTransformer
        self.chunks = chunks
        self.emb = SentenceTransformer(EMB, device=device)
        self.rr = CrossEncoder(RERANK, device=device)
```
(the rest of `__init__` is unchanged).

- [ ] **Step 2: `prepare_dir` takes an output path**

In `src/gatekeep/cli.py`, replace `prepare_dir` with:
```python
@command
def prepare_dir(folder, out="data/chunks.json"):
    """Parse every .html in a folder; chapter = file name. Writes the chunks to `out` (default data/chunks.json)."""
    from gatekeep import corpus
    chunks = []
    for p in sorted(Path(folder).glob("*.html")):
        for c in corpus.chunk(corpus.parse(str(p)), tag=p.stem):
            chunks.append({**c, "id": len(chunks)})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(chunks, open(out, "w"))
    print(len(chunks), "chunks from", len({c["chapter"] for c in chunks}), "pages ->", out)
```

- [ ] **Step 3: The `web` command**

In `src/gatekeep/cli.py`, add above the `# --- new commands go above this line ---` marker:
```python
WEB_CORPORA = {  # name: (label, chunks file, how to prepare it, example questions)
    "book": ("Book", "data/chunks.json", "python -m gatekeep.cli prepare <your copy of the book>.pdf",
             ["What is the difference between bagging and boosting?",
              "Why do we scale features before training an SVM?",
              "What does the learning rate do in gradient descent?"]),
    "sklearn": ("scikit-learn docs", "data/sk/chunks.json",
                "python scripts/fetch_sklearn_docs.py, then python -m gatekeep.cli prepare_dir data/sk data/sk/chunks.json",
                ["What is the difference between bagging and boosting?",
                 "How is the silhouette coefficient used to evaluate clustering?",
                 "How do I fine-tune a large language model with LoRA?"]),
}


@command
def web(port="8000"):
    """Local web app on http://127.0.0.1:<port>: Fast (all Laya) or Careful (Laya, then Gemma), book or scikit-learn docs."""
    import webbrowser
    from gatekeep import demo
    from gatekeep.corpus import Index
    from gatekeep.gates import LayaGate
    from gatekeep.llm import LLM
    from gatekeep.web import make_server
    for f, fix in (("models/ckpts.json", "copy notebook 10's models/ folder into the project"),
                   ("results/tau.json", "it comes from the evaluation run (notebook 11)")):
        if not Path(f).exists():
            raise SystemExit(f"{f} is missing: {fix}")
    llm, taus = LLM(), json.load(open("results/tau.json"))
    laya = LayaGate.load(json.load(open("models/ckpts.json")))  # names any missing weights or calibration file
    corpora = {}
    for name, (label, path, prepare, examples) in WEB_CORPORA.items():
        print("building the", label, "index" if Path(path).exists() else "index: skipped, not prepared")
        corpora[name] = {"label": label, "prepare": prepare, "examples": examples,
                         "index": Index(load_chunks(path), device="cpu") if Path(path).exists() else None}
    srv = make_server(corpora, lambda q, index, mode: demo.ask_events(q, index, llm, laya, taus, mode), int(port))
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    print("GateKeep is running at", url, "(Ctrl+C to stop)")
    webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
```

- [ ] **Step 4: Run the tests**

Run: `.\.venv\Scripts\python.exe -m pytest -q` → all pass (nothing else calls `prepare_dir` with positional output, and `Index`'s new argument has a default).

- [ ] **Step 5: Commit**

```powershell
git add src/gatekeep/corpus.py src/gatekeep/cli.py
git commit -m "feat: web command; docs corpus in its own file; indexes can stay on the CPU"
```

---

### Task 5: Prepare the docs, try it in a browser, document it

**Files:**
- Modify: `README.md` (add a "Web app" section before "Reproducing")
- Writes private, git-ignored `data/sk/` (the downloaded pages and `data/sk/chunks.json`).

- [ ] **Step 1: Prepare the scikit-learn corpus (once)**

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\fetch_sklearn_docs.py
.\.venv\Scripts\python.exe -m gatekeep.cli prepare_dir data/sk data/sk/chunks.json
```
Expected: `22 pages in data/sk`, then about `1521 chunks from 22 pages -> data/sk/chunks.json`. `data\chunks.json` (the book) is unchanged.

- [ ] **Step 2: Start the app**

```powershell
$env:OLLAMA_API_KEY = [Environment]::GetEnvironmentVariable('OLLAMA_API_KEY','User'); $env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe -m gatekeep.cli web
```
Run it in the background. Expected: two "building the ... index" lines (a minute or two each on the CPU), then `GateKeep is running at http://127.0.0.1:8000`.

- [ ] **Step 3: Check it in a browser**

Open `http://127.0.0.1:8000` (in the built-in browser pane or Chrome) and check, taking a screenshot of each:
1. The welcome screen: Kalam heading with the underline, three example questions, both corpora selectable, Fast selected.
2. Book + Fast: click "What is the difference between bagging and boosting?". The trace fills in step by step with blue **Laya** badges and times; an answer appears with source chips; clicking `[1]` shows the passage with section and page.
3. Book + Careful: the same question; at least one grey **Gemma** badge in the grade row.
4. scikit-learn docs + Fast: "How do I fine-tune a large language model with LoRA?" ends in a refusal with a reason.
5. A phone-width window (375 px): the sidebar becomes a top bar and nothing overflows sideways.
6. "New chat" clears the conversation and shows the welcome screen again.
Fix anything broken in the Task 2 files, re-run the full test suite, and commit the fix with a message that names it.

- [ ] **Step 4: Add the README section**

Insert before the "Reproducing" section of `README.md` (keep the rest unchanged):
```markdown
## Web app

A local page for asking questions and watching each gate decide. It runs on your PC only (`127.0.0.1`) and needs the v2 weights in `models/`, `results/tau.json` and your Ollama key.

```powershell
$env:OLLAMA_API_KEY = [Environment]::GetEnvironmentVariable('OLLAMA_API_KEY','User'); $env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe -m gatekeep.cli web
```

Fast mode uses Laya for every gate (V2, about 1.5 LLM calls per answer). Careful mode lets Laya decide first and asks Gemma when Laya is unsure (V3, about 8 calls). The book corpus uses your own copy of the book; the scikit-learn docs need a one-time `python scripts/fetch_sklearn_docs.py` and `python -m gatekeep.cli prepare_dir data/sk data/sk/chunks.json`.
```

- [ ] **Step 5: Commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git add README.md
git commit -m "docs: how to run the local web app"
```

---

## Self-review notes

- Spec coverage: `ask_events` and the rebuilt `ask` (Task 1); page, colours, fonts, sidebar, welcome, turns, chips, live trace, refusals, input bar, footer note, phone layout, in-memory history (Task 2); server routes, NDJSON, 127.0.0.1, lock, 400s, error events (Task 3); `web` command, missing-file messages, not-ready corpus, separate docs file, GPU memory (indexes on CPU) (Task 4); manual browser pass and README (Task 5). Out-of-scope items have no task.
- Placeholder scan: every code step carries its code; manual checks list concrete actions.
- Names used across tasks: `ask_events(question, index, llm, laya, taus, mode)`, `who`, `gate_line`, `_gates`, `make_server(corpora, ask, port)`, corpus dict keys `label`/`index`/`prepare`/`examples`, event `type`s `step`/`gate`/`passages`/`answer`/`done`/`error`, `WEB_CORPORA`, `Index(chunks, device)`, `prepare_dir(folder, out)`.
