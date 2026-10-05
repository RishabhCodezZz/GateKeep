from gatekeep.eval_gates import dev_acc_from_csv, evaluate_gate, hit_at_k, retrieval_recall
from gatekeep.gates import Gate


class AlwaysYes(Gate):
    def decide_many(self, gate, texts):
        return [("yes", 0.9)] * len(texts)


def test_evaluate_gate_metrics():
    rows = [{"text": "a", "label": "yes"}, {"text": "b", "label": "no"}, {"text": "c", "label": "yes"}]
    r = evaluate_gate(AlwaysYes(), "grade", rows)
    assert abs(r["acc"] - 2 / 3) < 1e-9 and r["ok"] == [True, False, True] and r["conf"] == [0.9] * 3
    assert r["p50_ms"] >= 0 and 0 <= r["ece"] <= 1


def test_evaluate_gate_reports_recall_of_yes_for_the_grade_gate():
    rows = [{"text": "a", "label": "yes"}, {"text": "b", "label": "yes"}, {"text": "c", "label": "no"}]

    class SaysYesOnlyToA(Gate):
        def decide_many(self, gate, texts):
            return [("yes", 0.9) if t == "a" else ("no", 0.9) for t in texts]

    r = evaluate_gate(SaysYesOnlyToA(), "grade", rows)
    assert r["recall_yes"] == 0.5 and r["precision_yes"] == 1.0  # it rejects half the relevant passages


class Idx:
    def search(self, q, k=20):
        return [{"id": i, "text": f"t{i}"} for i in range(8)]

    def rerank(self, q, hits, k=5):
        return hits[:k]  # the cross-encoder finds ids 0-4


class LayaPrefersLast(Gate):
    def decide_many(self, gate, texts):
        return [("yes", 0.99) if t.endswith("t7") else ("no", 0.9) for t in texts]


def test_hit_at_k_compares_the_two_rankers():
    items = [{"q": "q", "gold": [7], "answerable": True}, {"q": "q", "gold": [], "answerable": False}]
    assert hit_at_k(Idx(), LayaPrefersLast(), items, k=5) == {"cross_encoder": 0.0, "laya": 1.0}


def test_retrieval_recall_separates_search_misses_from_rerank_misses():
    items = [{"q": "q", "gold": [2], "answerable": True},      # found by search and kept by the reranker
             {"q": "q", "gold": [7], "answerable": True},      # found by search (top 8) but dropped by the reranker
             {"q": "q", "gold": [99], "answerable": True},     # never retrieved
             {"q": "q", "gold": [], "answerable": False}]
    r = retrieval_recall(Idx(), items)
    assert r == {"n": 3, "in_search_top20": 2 / 3, "in_rerank_top5": 1 / 3}


def test_dev_acc_from_csv_reads_a_backends_dev_accuracy_for_one_gate(tmp_path):
    p = tmp_path / "gates.csv"
    lines = ["gate,backend,split,n,f1,acc,ece,p50_ms,p95_ms,recall_yes,precision_yes",
             "grade,llm,dev,340,0.8,0.81,0.1,400,900,0.7,0.8",
             "grade,llm,test,963,0.8,0.83,0.1,400,900,0.7,0.8",
             "route,llm,dev,130,0.5,0.88,0.1,400,900,,"]
    p.write_text("\n".join(lines) + "\n")
    assert dev_acc_from_csv(str(p), "grade", "llm") == 0.81
    assert dev_acc_from_csv(str(p), "route", "llm") == 0.88


def test_evaluate_gate_reports_recall_for_every_label():
    rows = [{"text": "a", "label": "x"}, {"text": "b", "label": "x"}, {"text": "c", "label": "y"}]

    class Guess(Gate):
        def decide_many(self, gate, texts):
            return [("x", 0.9) if t in ("a", "b") else ("x", 0.6) for t in texts]  # never says y

    assert evaluate_gate(Guess(), "route", rows)["per_label_recall"] == {"x": 1.0, "y": 0.0}


def test_save_rows_replaces_rows_with_the_same_key_instead_of_duplicating(tmp_path):
    from gatekeep.eval_gates import save_rows
    p = str(tmp_path / "g.csv")
    header = ["gate", "backend", "split", "acc"]
    save_rows(p, header, [{"gate": "grade", "backend": "llm", "split": "test", "acc": 0.8},
                          {"gate": "grade", "backend": "sklearn", "split": "test", "acc": 0.6}])
    save_rows(p, header, [{"gate": "grade", "backend": "llm", "split": "test", "acc": 0.9}])
    import csv
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 2 and {r["backend"]: r["acc"] for r in rows} == {"llm": "0.9", "sklearn": "0.6"}
