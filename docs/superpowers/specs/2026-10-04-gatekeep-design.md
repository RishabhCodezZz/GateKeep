# GateKeep: agentic RAG design

Date: 2026-10-04 · Historical design document. The code, `results/` and the README are the source of truth where they differ.

Current project context: interest in typed-decision models such as [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) motivated the use of open-source [Laya](https://huggingface.co/convaiinnovations/laya) for GateKeep's four gates. The original scope and benchmark criteria below are retained as a record of the experiment design.

## Question

GateKeep uses fine-tuned Laya gates inside an agentic RAG graph to judge passage relevance, answer grounding and other workflow decisions. The original design explores when Laya (421M, non-autoregressive, ~33 ms) can make these decisions and when it should hand them to an LLM. Its planned deliverables include the implementation, measured comparisons, tables and plots. The current project also includes a local web app; the README describes that interface.

## Original scope

- Use accessible models and APIs; Jev was waitlisted with a closed API at design time. The original delivery plan is a public repository, README plots and a 90 s screen recording, with hosting outside that scope.
- Build on classifier-gating approaches such as Adaptive-RAG and Corrective RAG and Laya's `LayaRouter` for LangGraph. The focus is integrating Laya across four gates and measuring their behavior.
- Use a cross-encoder as the primary reranker. Evaluate Laya as a reranker in a separate ablation.

## Corpus

Primary: Géron, *Hands-On ML with Scikit-Learn, Keras & TensorFlow* (2nd ed.). Parsed once with Docling, chunked with Docling's HybridChunker (tokenizer = bge-small, `max_tokens=300`, because Laya reads ~512 tokens and G2 input is question + chunk). Each chunk keeps `section` (Docling heading path) and `page`; its `chapter` comes from the PDF bookmarks by page (Docling's headings are a flat list of section titles, not chapters).
The book is copyrighted: it and everything derived from it stay in private storage (git-ignored `data/`, `cache/`; private HF dataset repo for persistence between Kaggle sessions). The public repo ships code and aggregate results only.
Out-of-distribution corpus: the scikit-learn user guide (HTML pages; license to be confirmed on first fetch, believed BSD-3).

## Pipeline (LangGraph)

```
question → G1 route ─ direct/out_of_scope → answer or refusal
              │ retrieve
   FAISS top-20 (bge-small) → cross-encoder rerank top-5
   G2 grade each chunk → none relevant → rewrite query → retry (max 2) → else refuse
   Gemma writes answer
   G3 grounded? and G4 sufficient? → either "no" → regenerate (max 2, temp 0.7) → else flag
```

| Gate | Type | Input text |
|---|---|---|
| G1 route | choice: retrieve / direct / out_of_scope | question |
| G2 grade | yes/no per chunk | question + chunk |
| G3 grounded | yes/no | evidence (≤1200 chars) + answer |
| G4 sufficient | yes/no | question + answer |

Every gate is answered by an interchangeable backend with one interface: `decide_many(gate, texts) -> [(label, confidence)]`.
Backends: `LLMGate` (Gemma, thinking off), `SkGate` (TF-IDF + logistic regression baseline), `LayaGate` (zero-shot or fine-tuned), `Cascade(fast, slow, tau)`.

Variants: V0 no gates · V1 all gates by Gemma · V2 all gates by Laya · V3 Laya→Gemma cascade.
Multi-agent ablation: flat graph (blind retry) vs Researcher / Writer / Critic subgraphs where the Critic passes its reason to the Writer. The measured difference is the feedback channel; the subgraph split itself is structure.

## Models (via Ollama Cloud with the author's API key; NVIDIA NIM is a switchable fallback)

- Gemma 4 31B (`gemma4:31b` on Ollama): answer writer and baseline LLM judge for the gate comparisons.
- Nemotron 3 Super (`nemotron-3-super` on Ollama): labeler for training data and grader of final answers, from a different model family than the writer. Selected after measuring quota on 2026-10-04: Ultra cost ~0.00018 of the monthly quota and 8.9 s per call, Super ~0.00004 and 1.9 s, Gemma ~0.00002 and 0.4 s.
- Nemotron 3 Ultra: only validates Super. About 200 grading calls are repeated with Ultra and the agreement rate is reported.
- Laya: `convaiinnovations/laya`, typed-decisions checkpoint as the fine-tuning base; one fine-tuned copy per gate on Kaggle 2×T4.

## Data and splits

- Questions are generated from chunks (Super), single-chunk and two-chapter multi-hop; the source chunk is the gold passage. Unanswerable questions target topics absent from the book (guarded by a substring check).
- Use separate train / dev / test chapters to account for material repeated within the book. Unanswerable topics are split the same way.
- The author hand-writes 50–100 questions (evaluation only; reported separately because they are not chapter-held-out) and hand-checks 50 generated items.
- G3 training data uses real Gemma answers from retrieved passages, generated at temperatures 0.0 and 0.7 and labeled by Super. Dev and test rows use temperature 0.0. This replaces the first design's faithful-versus-corrupted answers, whose writing style revealed the class. Real-hallucination data (RAGTruth) is an optional extension for further G3 evaluation.
- Dev set picks the cascade threshold per gate; test is touched once per reported number.

## Evaluation

Per gate: macro-F1, ECE, p50/p95 latency per decision, coverage-vs-accuracy curve; backends LLM / sklearn / Laya zero-shot / Laya fine-tuned. Plus hit@5 for cross-encoder vs Laya-as-reranker.
End to end (V0–V3, ~300 answerable + ~32 unanswerable test questions, handwritten set separately): correctness and faithfulness (Super), LLM calls per question, gate LLM calls, latency, 95% bootstrap CIs. Latency of cached LLM calls replays the original measured latency.
OOD: rerun V1 and V3 on the scikit-learn corpus with the book-trained gates.

## Benchmark criteria (fixed before any run)

Primary: V3 correctness within 2.5 points of V1 and at least 5× fewer gate-level LLM calls than V1.
The report records the behavior of every gate, including cases needing LLM fallback and tasks handled by the sklearn baseline. Those comparisons inform which backend to use for each decision.

## Deliverables

GitHub repo (code, results CSV/PNG, README with the plots), write-up in the README, 90 s screen recording, résumé lines with the measured numbers.

The public Kaggle demo notebook is deferred at the author's request. Readers can reproduce the experiments with the repository notebooks, using their own corpus, API credentials and saved backups. The demo is not required for the current delivery.

## Implementation checks

1. Evaluate G3/G4 within their input window and use the cascade for decisions that need an LLM.
2. Check Docling's code, equation and chapter parsing against PDF bookmarks; PyMuPDF4LLM is the planned fallback.
3. Review 50 generated items by hand and include human-written questions alongside LLM labels.
4. Cache calls, pin model IDs and log versions to support reruns across free-tier limits and model updates. The original plan measures Ollama Cloud quota use in Task 1, with NIM as a fallback.
5. Persist `data/ cache/ results/ models/` in private storage between Kaggle sessions.
6. Verify Laya's fine-tuning API and noul answer encoding in Tasks 1 and 8 before use.

## Open items

Injection-shield gate stays a stretch goal for the buffer weeks.

## v2 addendum (2026-10-07, written before any v2 run)

v2 incorporates five refinements identified in the first round: a router task separating machine-learning questions from off-topic messages, replacing coverage judgments based on 48 out-of-scope questions about 6 topics; separate training and calibration rows; soft targets replacing hard 0/1 targets; an accuracy floor for thresholds beyond the original 0.5 router setting; and a larger evidence window than the original 1,200-character cap for checkpoints that read 1,024 tokens. The original calibration slice included copied training rows; v2 fits calibration on the separate dev split. The changes and measurements are documented in docs/RESULTS.md.

Criteria, fixed now:
1. Primary, unchanged from v1: V3 correctness within 2.5 points of V1 and at least 5x fewer gate-level LLM calls than V1, on the same 343-question test set.
2. Router: on the route test split, recall of `retrieve` is at least 0.97 and recall of `off_topic` at least 0.80.
3. Fewer wrong refusals: V3 refuses at most 10 of the 311 answerable test questions (v1: 19).
v1 results stay in `results/v1/` and are reported next to v2, whatever v2 shows.
