from gatekeep.eval_gates import evaluate_gate, hit_at_k, retrieval_recall
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
