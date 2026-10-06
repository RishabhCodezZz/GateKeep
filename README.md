# GateKeep

An agentic RAG pipeline asks an LLM to judge whether a passage is relevant, whether an answer is supported by it and whether the answer addresses the question. GateKeep measures how well a small fine-tuned model, Laya (421M parameters, about 46 ms per decision), handles those judgments and when it should pass a decision to the LLM.

## Results

A cascade that lets fine-tuned Laya decide first and asks Gemma when Laya is unsure (V3) scores 91.3% on answer correctness, compared with 90.1% for the all-LLM-gates pipeline (V1), on 343 questions from held-out chapters. It uses 4.7 times fewer gate-level LLM calls, falling short of the 5 times target fixed before the runs. The pre-registered verdict is therefore a fail on cost. At the lower confidence threshold of tau 0.7, V3 uses 13.7 times fewer gate-level LLM calls and scores 90.4%, close to V1's 90.1%. Skipping the gates entirely (V0) gives the highest correctness, 95.6%. On faithfulness, V0 scores lowest overall, but that gap comes from the 32 unanswerable questions; on answerable questions all four variants are within each other's intervals. In these runs the gates cost some correctness and showed no measurable faithfulness gain.

## Setup

The main corpus is Géron's *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow* (2nd edition). Docling parses the PDF once and divides it at headings into chunks of at most 300 tokens, using the bge-small tokenizer. The chunk size leaves room for a question alongside the passage in Laya's roughly 512-token input window. Chapter assignments come from the PDF bookmarks and page numbers. The book is copyrighted, so the PDF and its derived text, chunks, questions, cached model replies and generated answers stay out of this repository. The repository contains code and aggregate results. Reproducing the book experiments requires your own copy of the PDF.

The pipeline runs in LangGraph. Its router chooses retrieval, a direct answer or an out-of-scope refusal. Retrieval takes the top 20 passages from a FAISS index using bge-small embeddings, then a cross-encoder reranks them to 5. The grade gate checks each passage. If it rejects them all, the system rewrites the query and retries up to twice before refusing. Gemma writes an answer from the surviving passages. The grounded and sufficient gates then check it; a failed check triggers up to two regenerations, after which the answer is flagged if it still fails.

| Gate | Decision | Input |
|---|---|---|
| route | retrieve, direct or out of scope | question |
| grade | relevant or not, per passage | question and passage |
| grounded | supported by the passages or not | evidence (up to 1,200 characters) and answer |
| sufficient | answers the question or not | question and answer |

All gates share an interface, so the pipeline can use Gemma as an LLM judge, a TF-IDF plus logistic regression baseline, zero-shot Laya, fine-tuned Laya or the cascade.

V0 has no gates; V1 uses Gemma for every gate; V2 uses fine-tuned Laya. V3 combines Laya with Gemma, using a confidence threshold for each gate. The thresholds are chosen on the dev set so that the decisions Laya accepts meet the LLM judge's dev accuracy: route 0.5, grade 0.83, grounded 0.69 and sufficient 0.5.

Gemma 4 31B writes answers and handles the baseline LLM gates, providing a mid-size judge for the comparison. Nemotron 3 Super labels training data and grades final answers, so the grader and writer come from different model families. Nemotron 3 Ultra checks Super's grading. Laya uses the typed-decisions checkpoint from `convaiinnovations/laya`, fine-tuned separately for each gate. The LLM calls go through Ollama Cloud and are cached. Fine-tuning ran on Kaggle T4 GPUs.

Generated questions use either one passage or passages from two chapters for multi-hop questions. Their source chunks are the gold passages. Unanswerable questions cover topics absent from the book. Because the book repeats material, the train, dev and test sets are split by chapter. The test set contains 343 questions: 311 answerable and 32 unanswerable. Results for 80 hand-written questions are reported separately because those questions are not restricted to held-out chapters. The dev set determines the thresholds; the test set is used to report results.

## Per-gate results

These are test-set results. F1 is macro-F1 and ECE is expected calibration error. Latency is measured per decision: p50 is the median, and p95 is the 95th percentile. The gate test sets contain 373 route examples, 963 grade examples, 311 grounded examples and 622 sufficient examples. Every router backend uses the same 373 examples: 311 answerable questions, 32 unanswerable questions and 30 direct/chit-chat messages.

| Gate | Backend | F1 | ECE | p50 ms | p95 ms |
|---|---|---|---|---|---|
| route | Gemma | 0.646 | 0.115 | 368 | 1257.4 |
| route | TF-IDF baseline | 0.638 | 0.060 | 0.8 | 1.0 |
| route | Laya zero-shot | 0.602 | 0.309 | 35 | 39.2 |
| route | Laya fine-tuned | 0.875 | 0.050 | 46 | 59.2 |
| grade | Gemma | 0.825 | 0.173 | 424 | 925.5 |
| grade | TF-IDF baseline | 0.613 | 0.043 | 1.1 | 1.3 |
| grade | Laya zero-shot | 0.583 | 0.049 | 35 | 38.6 |
| grade | Laya fine-tuned | 0.803 | 0.077 | 47 | 52.8 |
| grounded | Gemma | 0.720 | 0.228 | 492 | 666.8 |
| grounded | TF-IDF baseline | 0.453 | 0.091 | 1.2 | 1.4 |
| grounded | Laya zero-shot | 0.641 | 0.037 | 36 | 43.1 |
| grounded | Laya fine-tuned | 0.757 | 0.032 | 47 | 57.8 |
| sufficient | Gemma | 0.953 | 0.047 | 555 | 951.9 |
| sufficient | TF-IDF baseline | 0.514 | 0.006 | 1.0 | 1.3 |
| sufficient | Laya zero-shot | 0.857 | 0.168 | 34 | 36.4 |
| sufficient | Laya fine-tuned | 0.969 | 0.009 | 46 | 49.2 |

Fine-tuned Laya matches or beats Gemma on route, grounded and sufficient. On grade, its accuracy is about 2 points lower: 80.4% against 82.7%. It is roughly ten times faster than the Gemma calls. Fine-tuning substantially improves route and grade compared with zero-shot Laya. The TF-IDF baseline is fast and, on route, close to Gemma (accuracy 0.887 against 0.885, F1 0.638 against 0.646), but it is far behind on grade, grounded and sufficient.

Two cautions about this table. The Gemma gate reports a confidence of 1.0 for every parsed reply, so its ECE is simply its error rate and says nothing about calibration. And the latencies come from different places: Laya ran on a Kaggle T4, Gemma ran through a network API, and the TF-IDF rows were re-timed on a laptop CPU, so the speed ratio is indicative, not a controlled comparison.

Laya's weak spot is the router's out-of-scope class, where recall is 0.56. Gemma's is 0.16.

Retrieval finds a gold passage in the top 20 for 94.2% of test questions and keeps one in the reranked top 5 for 91.3%. Laya as the reranker gives hit@5 of 91.6%, compared with the cross-encoder's 91.3%, effectively a tie.

## End-to-end results

The following results use the 343-question test set, with 95% bootstrap intervals. Super grades correctness and faithfulness.

Correctness includes every question. Faithfulness is scored only for responses with retrieved passages, excluding the pipeline's fixed refusal response. Responses without a score are left out of its mean and intervals, so each variant is scored on a different set of questions: V0 is scored on all 343, V1 on 285, V2 on 294 and V3 on 293. This definition applies to every faithfulness figure below.

| Variant | Correct | Faithful | LLM calls | Gate LLM calls | Latency (s) |
|---|---|---|---|---|---|
| V0 no gates | 0.956 [0.933, 0.977] | 0.691 [0.647, 0.741] | 1.0 | 0 | 0.9 |
| V1 Gemma gates | 0.901 [0.869, 0.930] | 0.796 [0.747, 0.839] | 10.8 | 9.4 | 6.1 |
| V2 Laya gates | 0.910 [0.881, 0.939] | 0.745 [0.694, 0.793] | 1.4 | 0 | 1.2 |
| V3 cascade | 0.913 [0.883, 0.942] | 0.782 [0.730, 0.826] | 3.4 | 2.0 | 2.1 |

V0's low faithfulness of 0.691 needs a closer look. V0 declines the 32 unanswerable questions in ordinary prose, and the grader counts those answers as correct (100% correct on unanswerable questions) but also as unsupported by the retrieved passages, so all 32 score 0 on faithfulness. The gated variants return the fixed refusal on those questions, which is not scored. Restricting to answerable questions removes the artifact:

| Variant | Faithful, answerable questions only | Questions scored |
|---|---|---|
| V0 no gates | 0.762 [0.711, 0.810] | 311 |
| V1 Gemma gates | 0.796 [0.751, 0.839] | 285 |
| V2 Laya gates | 0.750 [0.699, 0.798] | 292 |
| V3 cascade | 0.784 [0.736, 0.829] | 292 |

The four intervals overlap. On the 273 answerable questions that every variant answered, the scores are 0.813 (V0), 0.806 (V1), 0.766 (V2) and 0.795 (V3), again with overlapping intervals. These runs give no evidence that the gates make answers more faithful.

![Correctness against LLM calls per question](results/tradeoff.png)

![Accuracy against the share of decisions Laya answers alone](results/coverage.png)

The success criteria were fixed before any run: V3 must stay within 2.5 points of V1 on correctness while using at least 5 times fewer gate-level LLM calls. V3 scores 1.2 points above V1, but the call ratio is 9.41 / 2.00 = 4.7, so it fails the cost criterion by a small margin. That margin is within sampling noise. Resampling the 343 questions in pairs gives a call ratio of 4.7 with a 95% interval of [4.3, 5.2], and 11% of resamples reach 5. The correctness gap (V1 minus V3) is -1.2 points with an interval of [-3.5, +1.2], so V3 is not measurably worse than V1.

Changing the confidence threshold affects call counts much more than answer quality.

| tau | Correct | Faithful | Gate LLM calls | Ratio vs V1 |
|---|---|---|---|---|
| 0.7 | 0.904 | 0.783 | 0.69 | 13.7 |
| dev-picked (above) | 0.913 | 0.782 | 2.00 | 4.7 |
| 0.85 | 0.910 | 0.770 | 3.11 | 3.0 |
| 0.95 | 0.916 | 0.788 | 4.75 | 2.0 |

Tau 0.7 was tried after the result with dev-picked thresholds. Its result is exploratory; the pre-registered verdict uses the dev-picked thresholds.

The correctness intervals for V1, V2 and V3 overlap, so these results do not establish a correctness advantage among them. V0 scores clearly higher on correctness. Most of the gap comes from questions the gated variants refuse: V1 refuses 26 answerable questions, and V2 and V3 refuse 19 each (5 at the router, 14 after no passage survived grading). When the gated variants answer, they are right 94 to 96% of the time. The results do not support a general claim that gates improve quality.

The 80 hand-written questions give these separate results:

| Variant | Correct | Faithful |
|---|---|---|
| V0 | 0.913 [0.850, 0.963] | 0.512 [0.400, 0.613] (answerable only: 0.615 [0.492, 0.738]) |
| V3 | 0.888 [0.813, 0.950] | 0.655 [0.517, 0.776] (answerable only: 0.655 [0.517, 0.776]) |

On the hand-written questions, V3 scores slightly lower than V0 on correctness. The faithfulness difference shrinks to 0.615 against 0.655 once the 15 unanswerable questions are left out, well inside the intervals. With only 80 questions the intervals are wide.

Super and Ultra agree on correctness for 95.5% of 200 sampled V1 answers. This is raw agreement between two models from the same family on a set where most answers are correct, so it is weaker evidence than a human check.

## Multi-agent ablation

The flat graph retries without passing the failed check's reason to the answer writer. The three-agent version uses researcher, writer and critic subgraphs, with the critic passing its reason to the writer. This comparison measures the change in structure together with that feedback.

| Variant | Correct | Faithful | LLM calls |
|---|---|---|---|
| V1 flat | 0.901 [0.869, 0.930] | 0.796 [0.747, 0.839] | 10.76 |
| V1 agents | 0.904 [0.869, 0.933] | 0.793 [0.740, 0.839] | 10.74 |
| V3 flat | 0.913 [0.883, 0.942] | 0.782 [0.730, 0.826] | 3.36 |
| V3 agents | 0.918 [0.889, 0.948] | 0.795 [0.747, 0.840] | 3.36 |

The differences in correctness and faithfulness are within the reported intervals, and call counts are nearly identical. This experiment found no measurable improvement from the agent structure and critic feedback.

## Out-of-distribution result

V1 and V3 were rerun on the scikit-learn user guide (22 pages, 1,521 chunks), using the book-trained gates and the same thresholds. None of the gates had trained on this corpus. The test set contains 105 questions: 73 answerable and 32 unanswerable.

| Variant | Correct | Faithful | LLM calls | Gate LLM calls |
|---|---|---|---|---|
| V1 | 0.924 [0.876, 0.971] | 0.894 [0.818, 0.955] | 11.9 | 10.5 |
| V3 | 0.914 [0.857, 0.962] | 0.892 [0.815, 0.954] | 2.7 | 1.5 |

V3 stays within a point of V1 and uses 7.2 times fewer gate calls, exceeding the 5 times target. These runs show no drop on this sample, but three things limit what it shows. The split is by page, so the 73 answerable test questions come from only 6 of the 22 pages (320 of the 1,521 chunks). The 32 unanswerable questions come from the same topic prompts as the book's test set, so they are probably largely the same questions and test the router again, not the new corpus. And 30% of the questions are unanswerable, against 9% on the book, which raises correctness because refusing is the easy case. A claim about generalization would need more evidence.

## Limits

- LLMs supply the training labels and final grades. We hand-checked 50 generated questions and used that review to write the question filters.
- Super labels the grounded gate's training data and also grades final faithfulness with a similar prompt, so the grounded results share a judge with the grader.
- The grounded gate uses real Gemma answers from retrieved passages in all three splits. Super labels whether every claim is supported by the evidence. Training includes answers generated at temperatures 0.0 and 0.7; dev and test use 0.0. These labels depend on Super's judgments.
- The pipeline calls its gates one after another, so cascade latency is the sum of the steps.
- Questions are split by chapter, but the negative and evidence passages for the gate training data come from the whole book, including test chapters. This probably biases the results slightly against the gates, and I did not measure it.
- The direct (chit-chat) examples are split by generation order, so dev and test contain different kinds of messages from train, with only 10 and 30 examples respectively.
- The notebooks install unpinned packages and clone the latest code from the default branch, so reruns can differ from the recorded environment.
- Laya reads about 512 tokens and handles at most roughly 20 options. Those limits determined the chunk size and evidence cap.
- The models run through a free tier whose limits and versions can change. Every call is cached, and `results/env.txt` records the package versions used for the runs.
- The scikit-learn test set is small, and its share of unanswerable questions is high.
- Adaptive-RAG and Corrective RAG already use small classifiers as gates, and Laya ships a `LayaRouter` for LangGraph. This project measures how a recent model performs in that role.

## Reproducing

The unit tests run on a laptop with Python 3.12 and need no models or API keys. On Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m pytest -q
```

On Linux or macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest -q
```

The corpus integration test is skipped unless Docling and Transformers are installed.

The experiments run on Kaggle (internet on, an `OLLAMA_API_KEY` secret, one T4 GPU, two for notebook 04). Run the notebooks in `notebooks/` in order: 01 parses the corpus, 02 generates questions and runs the baseline, 03 builds the gate data, 04 fine-tunes Laya, 05 runs the main comparison, 06 produces the verdict, sweep and grader check, 07 runs the agent ablation, and 08 tests the scikit-learn corpus. After 01, each notebook restores the previous backup, and 05 to 08 also need notebook 04's saved output attached, because the backup holds the checkpoint paths but not the weights. You need your own copy of the book, your own API key and your own backups.

`requirements.txt` pins the full pipeline to the versions in `results/env.txt` (recorded on Python 3.13.15, with `torch==2.11.0+cu128` there and the CUDA build left to the runtime here). Notebook 04 also installs Datasets, PyArrow, Pandas, Accelerate and Tabulate in its own setup cell. Model revisions and API responses can change, so the pins do not guarantee identical results.

Keep the book, the generated questions, the per-question answer files (`results/*rows*.jsonl`), the model weights and the call cache private. They are git-ignored. The `report` command needs the answer files and checkpoints, so the public CSVs are not enough to regenerate every plot.

## Credits

scikit-learn documentation © The scikit-learn developers, BSD-3-Clause. The pages are downloaded at run time and are not included here.
