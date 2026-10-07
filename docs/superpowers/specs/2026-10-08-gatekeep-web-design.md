# GateKeep web app: design

Date: 2026-10-08. Approved section by section in chat by the author.

## Goal

A local website for asking questions to GateKeep and watching how each answer was checked. It runs on the author's PC, uses the fine-tuned v2 gates, answers from either the book or the scikit-learn docs, and shows a live, gate-by-gate trace next to every answer. It also serves as the set for the 90-second video (scikit-learn docs only).

## Decisions

| Topic | Decision |
|---|---|
| Where it runs | Locally, on `127.0.0.1` only. Uses the RTX 3050 and the user's Ollama key. Not public. |
| Pipeline | A switch between **Fast** (V2, every gate by Laya) and **Careful** (V3, Laya first, Gemma when Laya is unsure). Fast is the default. |
| Corpus | A switch between **Book** (private) and **scikit-learn docs** (BSD-3). The docs are parsed once into their own file. |
| Conversation | Chat-style history in the page. Each question is answered on its own; the page says so. No follow-up rewriting. |
| Server | Python standard library (`http.server`), no new packages. Answers stream back as newline-delimited JSON. |
| Front end | Plain HTML, CSS and JavaScript, no framework, no build step. |
| Fonts | Kalam for the app name and the welcome heading only; Inter for questions, answers and the trace; JetBrains Mono for timings and code. |
| Gradio demo | Kept as it is for Kaggle use (notebook 09). |

## Components

**`src/gatekeep/demo.py`** (extend)
- New `ask_events(question, index, llm, laya, taus, mode)`: a generator that runs one question and yields events as they happen. `mode` is `"fast"` (V2) or `"careful"` (V3).
- Event types (plain dicts with a `type` key):
  - `step`: `node` (route, retrieve, grade, rewrite, generate, check, refuse, direct) and a short `text`
  - `gate`: `gate`, `labels`, `n`, `who` (`"Laya"`, or `"Laya, then Gemma for k of n"`), `ms`
  - `passages`: list of `{text, section, page}` for the passages the answer was based on
  - `answer`: `text`, `refused` (bool), `reason` (`"off_topic"`, `"no_passage"` or `null`)
  - `done`: `seconds`
  - `error`: `message`
- The existing `ask()` is rebuilt on top of `ask_events` so the Gradio app keeps its output.

**`src/gatekeep/web.py`** (new, about 100 lines, standard library only)
- `GET /` serves `index.html`; `GET /static/<file>` serves the CSS and JS.
- `GET /api/info` returns the corpora (name, label, ready or not, the command to prepare it when not ready) and three example questions per corpus.
- `POST /api/ask` takes `{"question", "mode", "corpus"}` and streams events as `application/x-ndjson`, one JSON object per line.
- Binds to `127.0.0.1`. A lock serves one question at a time (one GPU).
- The backend (indexes, LLM client, Laya gates, thresholds) is passed in, so tests can use fakes.

**`src/gatekeep/cli.py`** (extend)
- New `web` command: loads the LLM client, the Laya gates from `models/ckpts.json` (with each gate's `calibration.json`), the thresholds from `results/tau.json`, and an index for each corpus whose chunks exist; starts the server on port 8000 and opens the browser.
- `prepare_dir(folder, out="data/chunks.json")` gains an output path, so the docs go to `data/sk/chunks.json` and never replace the book's `data/chunks.json`.

**`src/gatekeep/static/`** (new): `index.html`, `app.css`, `app.js`.

## Screen

- Colours: background `#FAF9F6`, text `#252525`, primary `#5268B3`. White panels, fine borders, softly rounded corners.
- **Sidebar:** "GateKeep" in Kalam; corpus switch (a corpus that is not ready is greyed out with its fix command); mode toggle "Fast (Laya)" / "Careful (Laya + Gemma)"; "New chat"; the note "Each question is answered on its own."
- **Welcome:** "What would you like to know?" in Kalam with one hand-drawn SVG underline, and three clickable example questions for the current corpus.
- **A turn:** the question (right-aligned); the answer in Inter at about 17px, line height 1.6; source chips `[1] [2] [3]` that expand to the passage with its section and page (a `[n]` written in the answer is clickable too); a collapsible "How this answer was checked" panel that fills in live, one row per step, with a **Laya** (blue) or **Gemma** (grey) badge and the time in JetBrains Mono, ending with the total time.
- **Refusals** use a muted style with a reason: "Not about machine learning" (router) or "No passage in the book answered this (query rewritten twice)" (grade).
- **Input bar** pinned at the bottom: Enter sends, Shift+Enter adds a line, disabled while an answer streams.
- **Footer note:** Careful makes about 8 Gemma calls per answer, Fast about 1.5.
- At phone width the sidebar folds into a top bar.
- History lives in the page's memory only; reload or "New chat" clears it. Nothing is written to disk.

## Data flow

1. Page load calls `/api/info` and renders the corpus switch and examples.
2. Send posts `{question, mode, corpus}` to `/api/ask` and reads the response stream line by line.
3. `step` and `gate` events add trace rows; `passages` fills the source chips; `answer` fills the answer; `done` closes the turn; `error` shows a red note on the turn.

## Errors

- Missing `models/ckpts.json`, weights or calibration files: `web` does not start and prints one line naming the missing file and the fix.
- A corpus without chunks: the server starts with the other corpus; the missing one is greyed out with its prepare command.
- Ollama errors (bad key, no network, rate limit) and any exception during an answer: an `error` event; the server keeps running.
- Empty question, unknown mode or corpus, invalid JSON: HTTP 400 with a message. The page also blocks empty questions.
- A second request during an answer waits on the lock; the page prevents it by disabling Send.
- GPU memory: the four gates fit in 4 GB (3.7 GB measured); a load failure stops `web` with the error and suggests closing other GPU apps.

## Testing

- `tests/test_demo.py`: `ask_events` with the existing fakes: event order; Fast mode never calls the LLM gate; Careful mode reports Gemma when Laya's confidence is under the threshold; refusals carry their reason; `ask()` keeps its trace text.
- `tests/test_web.py`: the real server on a free port in a background thread with a fake backend: `GET /` returns HTML; `/api/info` lists corpora; `POST /api/ask` streams valid JSON lines ending in `done`; bad input returns 400; a failing backend yields an `error` event.
- Manual pass in a browser on `localhost`: one question per mode and corpus, the live trace, the source chips, a phone-width window, screenshots. The JavaScript and visuals are checked this way, not by unit tests.

## Out of scope

Public hosting, logins, uploading new documents, follow-up questions that use earlier turns, saving chat history, dark mode, and V0/V1 modes.
