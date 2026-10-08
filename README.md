# GateKeep

GateKeep answers questions about two sources: Géron's *Hands-On Machine Learning* and the scikit-learn user guide. It exists to test one idea: can a small fine-tuned model make the judgment calls inside a RAG pipeline instead of a large LLM?

A multi-step RAG pipeline has to decide whether a question needs a lookup, whether each retrieved passage is relevant, whether the answer is backed by the passages, and whether it answers the question. We call these four decisions gates. Laya, a 421M-parameter model fine-tuned once per gate, makes them in about 35 ms each on a Kaggle T4. Gemma 4 31B writes the answers and steps in when Laya is unsure.

## What we found

On 343 questions from held-out chapters:

- Laya's gates match the all-Gemma gates on correctness (92.1% against 91.5%) and make no LLM calls for the gates.
- The cascade, where Gemma checks only the decisions Laya is unsure about, also scores 91.5%. But it cuts gate-level LLM calls only 1.3 times against a target of 5, so the cost goal failed.
- The router works. It sends 99% of book questions to retrieval and turns away all 40 off-topic test messages.
- Skipping the gates entirely scored highest, 95.6%. Most of the gap comes from the gated variants refusing 15 to 23 answerable questions.

Tables, plots, the v1 results and the full list of limits are in [docs/RESULTS.md](docs/RESULTS.md).

## The web app

A local page where you ask a question and watch each gate decide. Every question searches both sources at once, and each passage says which one it came from. Laya checks the passages first. If that finds nothing usable, Gemma takes over and judges every passage. You see only the final answer, with a line saying who made which checks.

### Setup

You need Python 3.12 and an Ollama Cloud API key. The book is copyrighted, so bring your own PDF; it and everything derived from it stay out of git. The fine-tuned Laya weights are not in the repository either. Notebooks 10 and 11 on Kaggle produce them, along with `results/tau.json`; put both next to the code (`models/` and `results/`).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Prepare the two sources once:

```powershell
.\.venv\Scripts\python.exe -m gatekeep.cli prepare "<your copy of the book>.pdf"
.\.venv\Scripts\python.exe scripts\fetch_sklearn_docs.py
.\.venv\Scripts\python.exe -m gatekeep.cli prepare_dir data\sk data\sk\chunks.json
```

### Run

```powershell
$env:OLLAMA_API_KEY = [Environment]::GetEnvironmentVariable('OLLAMA_API_KEY','User'); $env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe -m gatekeep.cli web
```

Starting takes 8 to 10 minutes while it builds the search index on the CPU. Then it opens http://127.0.0.1:8000. A normal answer takes 10 to 20 seconds on a 4 GB laptop GPU. If Laya fails and Gemma takes over, expect 30 to 45.

## Tests

The unit tests need no models or API keys, and `requirements-test.txt` is enough to run them.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Where things are

- `src/gatekeep/`: the pipeline, gates, CLI and web app (`static/` is the page).
- `notebooks/`: the Kaggle runs. 01 to 08 made the v1 results; 10 and 11 fine-tune and evaluate v2.
- `results/`: the published CSVs and plots, with the first round in `results/v1/`.
- `docs/`: `RESULTS.md` is the full write-up, and `superpowers/specs/` has the design with its pre-registered criteria.

## Limits

- LLMs supplied the training labels and the final grades, so the results rest on those judgments.
- Gemma runs on a free tier whose limits and model versions can change. Every call is cached.
- The scikit-learn test is small (105 questions, 30% unanswerable), so it says little about how well the gates generalise.
- The notebooks install unpinned packages, so a rerun can differ from the recorded environment.

## Credits

scikit-learn documentation © The scikit-learn developers, BSD-3-Clause, downloaded at run time and not included. Laya is by ConvAI Innovations (`convaiinnovations/laya` on Hugging Face).
