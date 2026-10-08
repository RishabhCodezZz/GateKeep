# GateKeep

An agentic RAG pipeline asks an LLM to judge whether a passage is relevant, whether an answer is supported by it and whether the answer addresses the question. GateKeep measures how well a small fine-tuned model, Laya (421M parameters, 35 to 42 ms per decision on a Kaggle T4), handles those judgments and when it should pass a decision to the LLM. This README reports v2, which fixed five mistakes found in v1 and re-ran the measurements. The v1 results are kept in `results/v1/`.

## Results

In v2, the cascade that lets fine-tuned Laya decide first and asks Gemma when Laya is unsure (V3) scores 91.5% on answer correctness, the same as the all-LLM-gates pipeline (V1), on 343 questions from held-out chapters. V1 makes 1.34 times as many gate-level LLM calls as V3, far short of the 5 times target fixed before the runs, so the primary criterion fails on cost (v1 reached 4.7 times). The router criterion is met: fine-tuned Laya's router recalls 0.990 of `retrieve` questions and 1.000 of `off_topic` messages, against targets of 0.97 and 0.80. The refusal criterion is not met: V3 refuses 23 of the 311 answerable questions, where the limit was 10 and v1 refused 19. Fine-tuned Laya on its own (V2) scores 92.1% with no gate-level LLM calls, which matches V1 (difference +0.6 points, 95% interval [-2.9, +3.8]). Skipping the gates entirely (V0) still gives the highest correctness, 95.6%. On answerable questions, the faithfulness intervals of all four variants overlap.

## What changed from v1

v1 had five mistakes. Each one, and its fix in v2:

1. The router had to decide from the question alone whether the book covered a topic, and its out-of-scope class was trained on 48 questions about 6 topics. It refused questions about Atari, TF-Agents and "What is machine learning". In v2 the router chooses between `retrieve`, `direct` and `off_topic`: any machine-learning question goes to retrieval, and the grade gate later decides whether a passage supports an answer. New training data adds short general machine-learning questions (retrieve) and non-ML questions (off_topic), and the messages are shuffled before the train, dev and test cut.
2. Oversampling copied rows into training that were also in the calibration slice: 25 of the 99 router calibration rows had identical copies in training, including all 13 out-of-scope and all 12 small-talk rows. v2 drops oversampling and refits calibration on the dev split, which is never trained on, with histogram binning for buckets that temperature scaling cannot fix.
3. Training used hard 0/1 targets, which drove the fitted temperatures of grade and sufficient to the clamp of 5.0. v2 trains on soft targets of 0.9 and 0.1, as the official recipe does; the training script now fits temperatures of 2.208 for grade and 1.047 for sufficient. The evaluation uses temperatures refit on dev instead: 0.606 for route (on its choice), and on the yes/no output 1.390 for grade, 1.556 for grounded and 0.542 for sufficient. Grade and sufficient also get histogram binning; route and grounded have fewer than 200 dev rows, too few for binning. The fitted values are in `results/calibration.json`.
4. The cascade threshold only had to match Gemma's dev accuracy, and Gemma's router was weak, so the router threshold was 0.5 and the router never asked Gemma. v2 adds a floor: the decisions Laya accepts on dev must reach at least 0.90 accuracy.
5. The grounded gate saw at most 1,200 characters of evidence, while the fine-tuned checkpoints read 1,024 tokens (768 for the state). v2 raises the cap to 2,000 characters and re-judges the grounded gate's labels with that evidence.

The criteria for v2 were added to the design spec (`docs/superpowers/specs/2026-10-04-gatekeep-design.md`, "v2 addendum") before any v2 run. The main numbers side by side:

| Measure | v1 | v2 |
|---|---|---|
| V1 correct | 0.901 [0.869, 0.930] | 0.915 [0.880, 0.945] |
| V2 correct | 0.910 [0.880, 0.939] | 0.921 [0.892, 0.948] |
| V3 correct | 0.913 [0.883, 0.942] | 0.915 [0.883, 0.945] |
| V1 gate LLM calls per question | 9.41 | 9.56 |
| V3 gate LLM calls per question | 2.00 | 7.16 |
| Gate-call ratio V1 / V3 | 4.70 [4.27, 5.18] | 1.34 [1.31, 1.37] |
| Answerable questions V3 refuses | 19 | 23 |
| Cascade thresholds (route, grade, grounded, sufficient) | 0.5, 0.83, 0.69, 0.5 | 0.5, 1.01, 0.8, 0.5 |

## Setup

The main corpus is Géron's *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow* (2nd edition). Docling parses the PDF once and divides it at headings into chunks of at most 300 tokens, using the bge-small tokenizer. The chunk size leaves room for a question alongside the passage in a 512-token input window. Chapter assignments come from the PDF bookmarks and page numbers. The book is copyrighted, so the PDF and its derived text, chunks, questions, cached model replies and generated answers stay out of this repository. The repository contains code and aggregate results. Reproducing the book experiments requires your own copy of the PDF.

The pipeline runs in LangGraph. Its router chooses retrieval, a direct answer or an off-topic refusal. Retrieval takes the top 20 passages from a FAISS index using bge-small embeddings, then a cross-encoder reranks them to 5. The grade gate checks each passage. If it rejects them all, the system rewrites the query and retries up to twice before refusing. Gemma writes an answer from the surviving passages. The grounded and sufficient gates then check it; a failed check triggers up to two regenerations, after which the answer is flagged if it still fails.

| Gate | Decision | Input |
|---|---|---|
| route | retrieve, direct or off topic | question |
| grade | relevant or not, per passage | question and passage |
| grounded | supported by the passages or not | evidence (up to 2,000 characters) and answer |
| sufficient | answers the question or not | question and answer |

All gates share an interface, so the pipeline can use Gemma as an LLM judge, a TF-IDF plus logistic regression baseline, zero-shot Laya, fine-tuned Laya or the cascade.

V0 has no gates; V1 uses Gemma for every gate; V2 uses fine-tuned Laya. V3 combines Laya with Gemma, using a confidence threshold for each gate. Each threshold is the lowest value at which the decisions Laya accepts on the dev set reach the higher of Gemma's dev accuracy and 0.90. If no threshold up to 0.99 gets there, the threshold is 1.01 and Laya never decides that gate alone. The v2 thresholds are route 0.5, grade 1.01, grounded 0.8 and sufficient 0.5.

Gemma 4 31B writes answers and handles the baseline LLM gates, providing a mid-size judge for the comparison. Nemotron 3 Super labels training data and grades final answers, so the grader and writer come from different model families. Nemotron 3 Ultra checks Super's grading. Laya is fine-tuned separately for each gate from the root `convaiinnovations/laya` checkpoint with the official typed-decisions recipe; the zero-shot baseline is the `typed-decisions` subfolder. The LLM calls go through Ollama Cloud and are cached. Fine-tuning and the v2 evaluation ran on Kaggle T4 GPUs.

Generated questions use either one passage or passages from two chapters for multi-hop questions. Their source chunks are the gold passages. Unanswerable questions cover topics absent from the book. Because the book repeats material, the train, dev and test sets are split by chapter. The test set contains 343 questions: 311 answerable and 32 unanswerable. Results for 80 hand-written questions are reported separately because those questions are not restricted to held-out chapters. The dev set determines the thresholds and the calibration; the test set is used to report results.

## Per-gate results

These are test-set results. F1 is macro-F1 and ECE is expected calibration error. Latency is measured per decision: p50 is the median, and p95 is the 95th percentile. The gate test sets contain 453 route examples, 963 grade examples, 311 grounded examples and 622 sufficient examples. Every router backend uses the same 453 examples: 383 `retrieve` (the 343 test questions, including the 32 unanswerable ones, and 40 short general machine-learning questions), 40 `off_topic` and 30 `direct`.

| Gate | Backend | F1 | ECE | p50 ms | p95 ms |
|---|---|---|---|---|---|
| route | Gemma | 0.940 | 0.013 | 668 | 1044.4 |
| route | TF-IDF baseline | 0.750 | 0.069 | 0.8 | 1.0 |
| route | Laya zero-shot | 0.886 | 0.428 | 35 | 37.7 |
| route | Laya fine-tuned | 0.982 | 0.016 | 35 | 37.7 |
| grade | Gemma | 0.825 | 0.173 | 424 | 925.5 |
| grade | TF-IDF baseline | 0.613 | 0.043 | 1.1 | 1.3 |
| grade | Laya zero-shot | 0.583 | 0.049 | 36 | 38.7 |
| grade | Laya fine-tuned | 0.792 | 0.019 | 36 | 38.3 |
| grounded | Gemma | 0.660 | 0.248 | 637 | 891.6 |
| grounded | TF-IDF baseline | 0.461 | 0.037 | 1.2 | 1.6 |
| grounded | Laya zero-shot | 0.605 | 0.047 | 42 | 51.4 |
| grounded | Laya fine-tuned | 0.699 | 0.047 | 42 | 52.1 |
| sufficient | Gemma | 0.953 | 0.047 | 555 | 951.9 |
| sufficient | TF-IDF baseline | 0.514 | 0.006 | 0.8 | 1.0 |
| sufficient | Laya zero-shot | 0.857 | 0.168 | 36 | 37.9 |
| sufficient | Laya fine-tuned | 0.973 | 0.007 | 36 | 38.3 |

Fine-tuned Laya scores higher than Gemma on route, grounded and sufficient. On grade it is lower: accuracy 79.2% against 82.7%. It is more than ten times faster than the Gemma calls. Fine-tuning improves every gate over zero-shot Laya, most of all grade. The TF-IDF baseline is fast but well behind on every gate.

The router changed the most. With the new classes, fine-tuned Laya recalls 1.000 of `off_topic` and 1.000 of `direct` messages and 0.990 of `retrieve` questions. Gemma's router also improved, to 1.000 on `off_topic`, so much of the gain comes from the easier task. The v1 and v2 route test sets differ (373 rows with an `out_of_scope` class in v1, 453 rows with `off_topic` in v2), so this comparison is between two versions of the task:

| Route, fine-tuned Laya | v1 | v2 |
|---|---|---|
| F1 | 0.875 | 0.982 |
| `retrieve` recall | 0.984 | 0.990 |
| Off-topic recall (`out_of_scope` in v1) | 0.562 | 1.000 |

Calibration of fine-tuned Laya, as ECE on the test set:

| Gate | v1 | v2 |
|---|---|---|
| route | 0.050 | 0.016 |
| grade | 0.077 | 0.019 |
| grounded | 0.032 | 0.047 |
| sufficient | 0.009 | 0.007 |

Grounded got slightly worse on calibration and lower on F1 (0.757 in v1, 0.699 in v2), but its test labels were re-judged with the longer evidence, so the two grounded results come from different tests. Gemma's grounded F1 also dropped, from 0.720 to 0.660.

Two cautions about the per-gate table. The Gemma gate reports a confidence of 1.0 for every parsed reply, so its ECE is simply its error rate and says nothing about calibration. And the latencies come from different places: Laya and the TF-IDF baseline ran on Kaggle, while Gemma ran through a network API, so the speed ratio is indicative and was not measured under controlled conditions.

Retrieval finds a gold passage in the top 20 for 94.2% of test questions and keeps one in the reranked top 5 for 91.3%. Laya as the reranker gives hit@5 of 92.0%, compared with the cross-encoder's 91.3% (v1 Laya: 91.6%).

## End-to-end results

The following results use the 343-question test set, with 95% bootstrap intervals. Super grades correctness and faithfulness. V0 has no gates, so v2 did not rerun it; its rows are the v1 rows.

Correctness includes every question. Faithfulness is scored only for responses with retrieved passages, excluding the pipeline's fixed refusal response. Responses without a score are left out of its mean and intervals, so each variant is scored on a different set of questions: V0 is scored on all 343, V1 on 292, V2 on 302 and V3 on 289. This definition applies to every faithfulness figure below.

| Variant | Correct | Faithful | LLM calls | Gate LLM calls | Latency (s) |
|---|---|---|---|---|---|
| V0 no gates | 0.956 [0.933, 0.977] | 0.691 [0.647, 0.741] | 1.0 | 0 | 0.9 |
| V1 Gemma gates | 0.915 [0.880, 0.945] | 0.805 [0.760, 0.849] | 10.9 | 9.6 | 5.9 |
| V2 Laya gates | 0.921 [0.892, 0.948] | 0.745 [0.695, 0.795] | 1.5 | 0 | 1.2 |
| V3 cascade | 0.915 [0.883, 0.945] | 0.806 [0.761, 0.855] | 8.5 | 7.2 | 4.7 |

V0's low faithfulness of 0.691 has the same cause as in v1. V0 declines the 32 unanswerable questions in ordinary prose, and the grader counts those answers as correct but also as unsupported by the retrieved passages, so all 32 score 0 on faithfulness. The gated variants mostly return the fixed refusal on those questions, which is not scored. Restricting to answerable questions removes the artifact:

| Variant | Faithful, answerable questions only | Questions scored |
|---|---|---|
| V0 no gates | 0.762 [0.717, 0.810] | 311 |
| V1 Gemma gates | 0.808 [0.759, 0.852] | 291 |
| V2 Laya gates | 0.760 [0.716, 0.814] | 296 |
| V3 cascade | 0.809 [0.764, 0.854] | 288 |

V1 and V3 score about 5 points above V0 and V2, but all four intervals overlap, so these runs give no clear evidence that the gates make answers more faithful.

![Correctness against LLM calls per question](results/tradeoff.png)

![Accuracy against the share of decisions Laya answers alone](results/coverage.png)

The primary criterion was fixed before any run and is unchanged from v1: V3 must stay within 2.5 points of V1 on correctness while using at least 5 times fewer gate-level LLM calls. V3 and V1 have the same correctness, with a paired gap of 0.0 points and a 95% interval of [-0.9, +0.9]. A paired interval between two variants comes from resampling the per-question differences over the 343 questions with `gatekeep.metrics.bootstrap_ci` (1,000 resamples). The call ratio is 9.56 / 7.16 = 1.34, with a paired interval of [1.31, 1.37], so V3 fails the cost half by a wide margin.

The cascade saved fewer calls than in v1 because v2's honest calibration keeps Laya's grade confidence near 0.85, below Gemma's grade accuracy on dev of 0.877. After soft targets, a dev-refit temperature of 1.39 and histogram binning, the highest grade bin that holds any dev decision maps to 0.851, because 74 of its 87 dev decisions were correct (`results/calibration.json`; the two bins above it are empty and keep their midpoints, 0.9 and 0.967). No threshold reaches Gemma's accuracy, so the grade threshold is 1.01 and Gemma makes every grade decision in V3. That would have happened without the 0.90 floor too. The floor could only have changed the grounded threshold (Gemma's dev accuracy 0.764, threshold 0.8), since Gemma reaches at least 0.96 on dev for route and sufficient. In v1 the grade threshold was 0.83, but v1's grade confidences were overconfident: its training temperature hit the clamp of 5.0. The grade gate runs once per passage, so V3 still makes 7.2 gate-level LLM calls per question against V1's 9.6.

V2 matches V1 on correctness: V2 minus V1 is +0.6 points with a paired interval of [-2.9, +3.8], which is not significant. V0 still scores clearly higher. Most of that gap comes from questions the gated variants refuse:

| Variant | Answerable refused | By the router | After no passage survived grading | Correct when answering |
|---|---|---|---|---|
| V0 no gates | 0 | 0 | 0 | 95.2% |
| V1 Gemma gates | 20 | 0 | 20 | 96.2% |
| V2 Laya gates | 15 | 4 | 11 | 94.6% |
| V3 cascade | 23 | 3 | 20 | 96.5% |

The new router classes cut router refusals of answerable questions (v1: 6 for V1, 5 each for V2 and V3). V3 now refuses 20 questions after grading, the same as V1, which fits the grade threshold of 1.01: in both variants Gemma makes every grade decision. All four variants answer the 32 unanswerable questions correctly. The results do not support a general claim that gates improve quality.

## Hand-written questions

The 80 hand-written questions (65 answerable, 15 unanswerable) give these separate results. V0 was not rerun; its row is from v1.

| Variant | Correct | Faithful |
|---|---|---|
| V0 | 0.913 [0.850, 0.963] | 0.512 [0.400, 0.613] (answerable only: 0.615 [0.492, 0.738]) |
| V1 | 0.938 [0.888, 0.975] | 0.772 [0.649, 0.877] (answerable only: the same) |
| V3 | 0.925 [0.863, 0.975] | 0.772 [0.649, 0.877] (answerable only: the same) |

V1 and V3 had 57 answers scored for faithfulness, all on answerable questions. V3 scores slightly below V1 on correctness; in v1, hand-written V3 scored 0.888 against V0's 0.913. These questions are not restricted to held-out chapters, and with only 80 questions the intervals are wide.

## Out-of-distribution result

V1 and V3 were rerun on the scikit-learn user guide (22 pages, 1,521 chunks), using the book-trained gates and the same thresholds. None of the gates had trained on this corpus. The test set contains 105 questions: 73 answerable and 32 unanswerable.

| Variant | Correct | Faithful | LLM calls | Gate LLM calls |
|---|---|---|---|---|
| V1 | 0.914 [0.867, 0.962] | 0.892 [0.815, 0.954] | 12.4 | 11.0 |
| V3 | 0.914 [0.867, 0.962] | 0.908 [0.831, 0.969] | 10.4 | 9.0 |

V3 matches V1 on correctness and uses 1.23 times fewer gate calls (v1: 7.2 times). The thresholds come from the book's dev set, so the grade threshold of 1.01 applies here too. Three things limit what this run shows. The split is by page, so the 73 answerable test questions come from only 6 of the 22 pages (320 of the 1,521 chunks). The 32 unanswerable questions come from the same topic prompts as the book's test set, so they are probably largely the same questions and test the router again on familiar material. And 30% of the questions are unanswerable, against 9% on the book, which raises correctness because refusing is the easy case. A claim about generalization would need more evidence.

## v1 results

These experiments were run in v1 only, with v1's router classes, gates and thresholds. Their files are in `results/v1/`.

### Threshold sweep

Changing the confidence threshold affected call counts much more than answer quality.

| tau | Correct | Faithful | Gate LLM calls | Ratio vs V1 |
|---|---|---|---|---|
| 0.7 | 0.904 | 0.783 | 0.69 | 13.7 |
| dev-picked (v1) | 0.913 | 0.782 | 2.00 | 4.7 |
| 0.85 | 0.910 | 0.770 | 3.11 | 3.0 |
| 0.95 | 0.916 | 0.788 | 4.75 | 2.0 |

Tau 0.7 was tried after the result with dev-picked thresholds. Its result is exploratory; the pre-registered verdict uses the dev-picked thresholds.

### Multi-agent ablation

The flat graph retries without passing the failed check's reason to the answer writer. The three-agent version uses researcher, writer and critic subgraphs, with the critic passing its reason to the writer. This comparison measures the change in structure together with that feedback.

| Variant | Correct | Faithful | LLM calls |
|---|---|---|---|
| V1 flat | 0.901 [0.869, 0.930] | 0.796 [0.747, 0.839] | 10.76 |
| V1 agents | 0.904 [0.869, 0.933] | 0.793 [0.740, 0.839] | 10.74 |
| V3 flat | 0.913 [0.883, 0.942] | 0.782 [0.730, 0.826] | 3.36 |
| V3 agents | 0.918 [0.889, 0.948] | 0.795 [0.747, 0.840] | 3.36 |

The differences in correctness and faithfulness are within the reported intervals, and call counts are nearly identical. This experiment found no measurable improvement from the agent structure and critic feedback.

### Grader check

Super and Ultra agree on correctness for 95.5% of 200 sampled v1 V1 answers. This is raw agreement between two models from the same family on a set where most answers are correct, so it is weaker evidence than a human check.

## Limits

- LLMs supply the training labels and final grades. We hand-checked 50 generated questions and used that review to write the question filters.
- Super labels the grounded gate's training data and also grades final faithfulness with a similar prompt, so the grounded results share a judge with the grader.
- The grounded gate uses real Gemma answers from retrieved passages in all three splits. Super labels whether every claim is supported by the evidence. Training includes answers generated at temperatures 0.0 and 0.7; dev and test use 0.0. These labels depend on Super's judgments.
- The grounded labels changed in v2: they were re-judged with 2,000 characters of evidence, so the v1 and v2 grounded results are not the same test.
- The pipeline calls its gates one after another, so cascade latency is the sum of the steps.
- Questions are split by chapter, but the negative and evidence passages for the gate training data come from the whole book, including test chapters. This probably biases the results slightly against the gates, and I did not measure it.
- The router's `direct` and `off_topic` test classes are small, with 30 and 40 examples.
- The notebooks install unpinned packages and clone the latest code from the default branch, so reruns can differ from the recorded environment.
- The chunk size was chosen for a 512-token window. The fine-tuned checkpoints read 1,024 tokens (768 for the state), which sets the 2,000-character evidence cap. Laya handles at most roughly 20 options.
- The models run through a free tier whose limits and versions can change. Every call is cached. `results/env.txt` records the package versions of the v1 Kaggle runs and is identical to `results/v1/env.txt`; the v2 notebooks install packages unpinned.
- The scikit-learn test set is small, and its share of unanswerable questions is high.
- Adaptive-RAG and Corrective RAG already use small classifiers as gates, and Laya ships a `LayaRouter` for LangGraph. This project measures how a recent model performs in that role.

## Web app

A local page for asking questions and watching each gate decide. It runs on your PC only (`127.0.0.1`) and needs the v2 weights in `models/`, `results/tau.json` and your Ollama key.

```powershell
$env:OLLAMA_API_KEY = [Environment]::GetEnvironmentVariable('OLLAMA_API_KEY','User'); $env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe -m gatekeep.cli web
```

There is one mode, "auto". Every question searches the book and the scikit-learn docs together (one index; each passage names its source) and runs the Fast path first. If Fast ends with no usable passage, or with an answer the grounded or sufficient gate still flags after retries, the page says so and runs the Careful path, and only that final answer is shown. An off-topic refusal is final and does not escalate.

Careful is V3: Laya decides first and Gemma steps in when Laya is unsure (about 8 LLM calls per answer); Gemma judges passages with the same lenient question as Fast's re-check. Fast is V2 with one addition: when Laya's grade gate rejects every retrieved passage, Gemma re-checks them with a more lenient question, "does the passage help answer any part of the question?", because the strict question fails every passage of a two-part question like "what is bagging and boosting?" (once per search, so a question with no answer can take up to 10 Gemma checks: a rewrite that finds the same passages stops the search). Without it, Fast mode refused broad and comparison questions the book covers, such as "What is the difference between bagging and boosting?", because Laya's grade gate learned to accept a passage only when it answers the question on its own. The measured V2 results above have no such re-check.

The book uses your own copy of the book. A source that is not prepared is skipped, and the other is searched alone. The scikit-learn docs need a one-time `python scripts/fetch_sklearn_docs.py` and `python -m gatekeep.cli prepare_dir data/sk data/sk/chunks.json`. On a 4 GB laptop GPU the four Laya gates nearly fill the memory, so an answer takes 10 to 20 seconds; on Kaggle's T4 a gate decision takes about 35 ms.

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

The experiments run on Kaggle (internet on, an `OLLAMA_API_KEY` secret, one T4 GPU, two for the fine-tuning notebooks). The v1 results come from the code at git tag `v1`, with notebooks 01 to 08 run in order: 01 parses the corpus, 02 generates questions and runs the baseline, 03 builds the gate data, 04 fine-tunes Laya, 05 runs the main comparison, 06 produces the verdict, sweep and grader check, 07 runs the agent ablation, and 08 tests the scikit-learn corpus. After 01, each notebook restores the previous backup, and 05 to 08 also need notebook 04's saved output attached, because the backup holds the checkpoint paths but not the weights.

The v2 results come from the main branch, where notebooks 10 and 11 replace 04 to 08. First rebuild the gate data with `python -m gatekeep.cli gatedata` and attach the four `data/gates/<gate>_train.jsonl` files to Kaggle as a private dataset. Notebook 10 fine-tunes the four gates on two T4 GPUs and needs no API key; run it with Save & Run All, or the weights are lost. Notebook 11 needs the backup, notebook 10's saved output and the API key; upload the backup as a new version of the dataset, holding the rebuilt `data/` from `gatedata` and the cache. It calibrates on dev, evaluates each gate, runs V1 to V3 and the hand-written set, writes the verdict and plots, and runs the scikit-learn corpus. The v2 numbers in this README come from one run of notebook 11. You need your own copy of the book, your own API key and your own backups.

`requirements.txt` pins the full pipeline to the versions in `results/env.txt` (recorded on Python 3.13.15, with `torch==2.11.0+cu128` there and the CUDA build left to the runtime here). Notebook 04 also installs Datasets, PyArrow, Pandas, Accelerate and Tabulate in its own setup cell. Model revisions and API responses can change, so the pins do not guarantee identical results.

Keep the book, the generated questions, the per-question answer files (`results/*rows*.jsonl`), the model weights and the call cache private. They are git-ignored. The `report` command needs the answer files and checkpoints, so the public CSVs are not enough to regenerate every plot.

## Credits

scikit-learn documentation © The scikit-learn developers, BSD-3-Clause. The pages are downloaded at run time and are not included here.
