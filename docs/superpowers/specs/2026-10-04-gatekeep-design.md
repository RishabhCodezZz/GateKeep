# GateKeep: measured agentic RAG — Design

Date: 2026-10-04 · Historical design document. The code, `results/` and the README are the source of truth where they differ.

## Question

In an agentic RAG graph, an LLM is normally called for every small judgment ("is this chunk relevant?", "is the answer grounded?"). Can a fine-tuned **Laya** (421M, non-autoregressive, ~33 ms) make those judgments instead, and when should it hand off to the LLM? The deliverable is a measured answer (table + plots), not a product.

## Non-goals

- No Jev (waitlist, closed API). No live hosting (public repo + README plots + 90 s screen recording).
- No claim of novelty: Adaptive-RAG and Corrective RAG already use small classifiers as gates, and Laya already ships a `LayaRouter` for LangGraph. The contribution is the measurement on a two-week-old model.
- Laya is not used as a retriever or primary reranker; a cross-encoder does that. Laya-as-reranker is a small ablation only.

## Corpus

Primary: Géron, *Hands-On ML with Scikit-Learn, Keras & TensorFlow* (2nd ed.). Parsed once with **Docling**, chunked with Docling's HybridChunker (tokenizer = bge-small, `max_tokens=300`, because Laya reads ~512 tokens and G2 input is question + chunk). Each chunk keeps `section` (Docling heading path) and `page`; its `chapter` comes from the PDF bookmarks by page (Docling's headings are a flat list of section titles, not chapters).
The book is copyrighted: it and everything derived from it stay in **private** storage (git-ignored `data/`, `cache/`; private HF dataset repo for persistence between Kaggle sessions). The public repo ships code and aggregate results only.
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

Variants: **V0** no gates · **V1** all gates by Gemma · **V2** all gates by Laya · **V3** Laya→Gemma cascade.
Multi-agent ablation: flat graph (blind retry) vs Researcher / Writer / Critic subgraphs where the Critic passes its reason to the Writer. The measured difference is the feedback channel; the subgraph split itself is structure.

## Models (via Ollama Cloud with the author's API key; NVIDIA NIM is a switchable fallback)

- Gemma 4 31B (`gemma4:31b` on Ollama): answer writer and the baseline LLM judge (a realistic mid-size judge, so the comparison is not a strawman).
- Nemotron 3 Super (`nemotron-3-super` on Ollama): labeler for training data and grader of final answers (different family from the writer, so no self-grading). Chosen over Ultra after measuring quota on 2026-10-04: Ultra cost ~0.00018 of the monthly quota and 8.9 s per call, Super ~0.00004 and 1.9 s, Gemma ~0.00002 and 0.4 s.
- Nemotron 3 Ultra: only validates Super. About 200 grading calls are repeated with Ultra and the agreement rate is reported.
- Laya: `convaiinnovations/laya`, typed-decisions checkpoint as the fine-tuning base; one fine-tuned copy per gate on Kaggle 2×T4.

## Data and splits

- Questions are generated from chunks (Super), single-chunk and two-chapter multi-hop; the source chunk is the gold passage. Unanswerable questions target topics absent from the book (guarded by a substring check).
- **Split by chapter, never randomly** (the book repeats itself): train / dev / test chapters. Unanswerable topics are split the same way.
- The author hand-writes 50–100 questions (evaluation only; reported separately because they are not chapter-held-out) and hand-checks 50 generated items.
- G3 training data uses real Gemma answers from retrieved passages, generated at temperatures 0.0 and 0.7 and labeled by Super. Dev and test rows use temperature 0.0. The first design used faithful answers against corrupted ones, but the writing style gave the class away, so it was replaced. Real-hallucination data (RAGTruth) is an optional extension only if G3 underperforms.
- Dev set picks the cascade threshold per gate; test is touched once per reported number.

## Evaluation

Per gate: macro-F1, ECE, p50/p95 latency per decision, coverage-vs-accuracy curve; backends LLM / sklearn / Laya zero-shot / Laya fine-tuned. Plus hit@5 for cross-encoder vs Laya-as-reranker.
End to end (V0–V3, ~300 answerable + ~32 unanswerable test questions, handwritten set separately): correctness and faithfulness (Super), LLM calls per question, gate LLM calls, latency, 95% bootstrap CIs. Latency of cached LLM calls replays the original measured latency.
OOD: rerun V1 and V3 on the scikit-learn corpus with the book-trained gates.

## Success criteria (fixed before any run)

Primary: V3 correctness within 2.5 points of V1 **and** at least 5× fewer gate-level LLM calls than V1.
Equally publishable outcomes: Laya fails on some gates (likely G3/G4), and the report says where and why; or a sklearn baseline is already enough on G1/G2.

## Deliverables

GitHub repo (code, results CSV/PNG, README with the plots), write-up in the README, 90 s screen recording, résumé lines with the measured numbers.

The public Kaggle demo notebook is deferred at the author's request. Readers can reproduce the experiments with the repository notebooks, using their own corpus, API credentials and saved backups. The demo is not required for the current delivery.

## Risks

1. Laya weak on G3/G4 (hard tasks, small window) → the cascade covers it; report honestly.
2. Docling mangles code/equations or chapter detection → compare with PDF bookmarks; fallback PyMuPDF4LLM.
3. LLM-made labels are not ground truth → hand-check 50, use human questions.
4. Free-tier limits or model changes → cache every call, pin model IDs, log versions. Ollama Cloud's free quota is unpublished and GPU-time based; Task 1 measures it before we commit, with NIM (40 req/min, no daily cap) as the fallback.
5. Kaggle sessions end → persist `data/ cache/ results/ models/` to a private HF dataset repo.
6. Laya API details taken from its README summary (fine-tune API, noul answer encoding) → verified in Tasks 1 and 8 before use.

## Open items

Injection-shield gate stays a stretch goal for the buffer weeks.
