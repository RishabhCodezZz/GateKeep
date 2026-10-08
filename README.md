# GateKeep

GateKeep is a RAG application for asking questions about documents, with fine-tuned Laya gates and a LangGraph workflow. It searches a machine-learning textbook and the scikit-learn user guide, generates answers from retrieved passages, and shows the sources and checks behind each response in a local web interface.

A RAG pipeline retrieves passages to help a model answer a question. GateKeep adds four checks: whether to retrieve, which passages are relevant, whether the answer is supported, and whether it addresses the question. Laya is a 421M-parameter model, fine-tuned separately for each check. Gemma 4 31B writes the answers and can handle decisions passed to it by the gates.

## Why Laya

Decision models such as [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) prompted my interest in using typed decisions inside AI workflows. I chose [Laya](https://huggingface.co/convaiinnovations/laya), an open-source model with an Apache 2.0 license, to build that decision layer in GateKeep. Its local inference and fine-tuning support let me train the four gates, calibrate their confidence and connect their outputs to the RAG workflow.

## Features

- A document pipeline with Docling parsing, FAISS search, cross-encoder reranking and source metadata. The web app searches Géron's *Hands-On Machine Learning* and the scikit-learn user guide together.
- Interchangeable gate backends: a TF-IDF classifier, LLM judges, zero-shot Laya, fine-tuned Laya and a confidence-based cascade.
- A fine-tuning and calibration workflow on Kaggle, with separate train, dev and test data and confidence thresholds selected on dev.
- Query rewriting and answer regeneration with bounded retries, plus a local demo that traces the decisions as they happen.

Four separately fine-tuned Laya models handle the gate decisions. Calibration and per-gate thresholds control when the cascade asks Gemma to make a decision. The repository includes the training notebooks, evaluation scripts and results from both rounds of development.

## Evaluation

The end-to-end comparison uses 343 questions from held-out chapters. Router figures come from the separate 453-example gate test:

- The pipeline with Laya handling all four gates scored 92.1% correctness, compared with 91.5% for Gemma gates. It made about 1.5 total LLM calls per question versus 10.9; Gemma still wrote the answers.
- Fine-tuned Laya scored higher macro-F1 than Gemma on routing, grounding and sufficiency. Its median gate latency was 35 to 42 ms on a Kaggle T4; Gemma timings include network API calls.
- The router sent 99% of retrieval examples to search and identified all 40 off-topic test messages.

These figures describe the evaluated Laya-gated configuration. The web app adds a passage re-check and automatic fallback for interactive use. Comparisons with the ungated baseline, cascade results, benchmark targets and evaluation scope are documented in [docs/RESULTS.md](docs/RESULTS.md).

## The web app

Ask a question and watch the retrieval and gate decisions as they happen. Every question searches both prepared sources together, and each passage names its source. Laya checks the passages first, with a Gemma re-check if all are rejected. If the first attempt finds no supporting passage or ends with a flagged answer, the app runs the cascade and displays its result. The final answer includes supporting passages and a trace of who performed the checks.

### Setup

You need Python 3.12 and an Ollama Cloud API key. The book is copyrighted, so bring your own PDF; it and everything derived from it stay out of git. The fine-tuned Laya weights are not in the repository either. Notebooks 10 and 11 on Kaggle produce them, along with `results/tau.json`; put both next to the code (`models/` and `results/`).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

Prepare the two sources once:

```powershell
.\.venv\Scripts\python.exe -m gatekeep.cli prepare "<your copy of the book>.pdf"
.\.venv\Scripts\python.exe scripts\fetch_sklearn_docs.py
.\.venv\Scripts\python.exe -m gatekeep.cli prepare_dir data\sk data\sk\chunks.json
```

### Run

```powershell
$env:OLLAMA_API_KEY = [Environment]::GetEnvironmentVariable('OLLAMA_API_KEY','User')
.\.venv\Scripts\python.exe -m gatekeep.cli web
```

On the tested laptop, startup takes 8 to 10 minutes while the CPU builds the search index. The app then opens http://127.0.0.1:8000. Answers typically take 10 to 20 seconds with a 4 GB laptop GPU, or 30 to 45 seconds when the app runs the cascade after the first attempt.

## Using and adapting GateKeep

Use the local app to explore machine-learning concepts and inspect the passages behind an answer. Each question runs independently. The gate trace is also useful for demonstrating how retrieval, filtering and answer checks fit together.

The shared gate interface lets you compare backends or study one gate at a time. To adapt the project to another domain, prepare its documents and update the machine-learning-specific router criteria and training data. The notebooks provide the workflow for fine-tuning, calibration and evaluation.

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

## Evaluation scope

- LLMs supplied the training labels and final grades. Their judgments define the reported scores; the detailed write-up explains the human checks and grader comparison.
- Gemma runs on a free tier whose limits and model versions can change. Every call is cached.
- The scikit-learn run explores transfer to a second source using 105 questions, 30% unanswerable. Broader transfer remains a direction for further evaluation.
- The notebooks install unpinned packages, so a rerun can differ from the recorded environment.

## Credits

scikit-learn documentation © The scikit-learn developers, BSD-3-Clause, downloaded at run time and not included. Laya is by ConvAI Innovations (`convaiinnovations/laya` on Hugging Face).
