from gatekeep.graph import REFUSAL
from gatekeep.report import breakdown


def row(answerable, answer="an answer", correct=1, route="retrieve", kept=3):
    return {"answerable": answerable, "answer": answer, "correct": correct, "route": route, "kept": kept}


def test_breakdown_separates_answerable_unanswerable_and_refusals():
    rows = [row(True), row(True, correct=0),                                   # answered: one right, one wrong
            row(True, REFUSAL, 0, route="out_of_scope", kept=0),               # refused by the router
            row(True, REFUSAL, 0, route="retrieve", kept=0),                   # refused: no passage survived grading
            row(False, REFUSAL, 1, route="out_of_scope", kept=0),              # correctly declined
            row(False, "made something up", 0)]                                # should have declined
    b = breakdown(rows)
    assert b["n"] == 6 and b["answerable"] == 4 and b["unanswerable"] == 2
    assert b["correct_answerable"] == 0.25 and b["correct_unanswerable"] == 0.5
    assert b["refused_answerable"] == 2 and b["refused_by_router"] == 1 and b["refused_after_grading"] == 1
    assert b["correct_when_answered"] == 0.5  # 1 of the 2 answerable questions it actually answered


def test_breakdown_handles_a_variant_with_no_refusals():
    b = breakdown([row(True), row(True)])
    assert b["refused_answerable"] == 0 and b["correct_when_answered"] == 1.0 and b["correct_unanswerable"] is None
