"""Reading the results: where a variant wins or loses, in numbers (verdict and plots are added in Task 10)."""
from gatekeep.graph import REFUSAL


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def breakdown(rows):
    """Split a variant's rows into answerable / unanswerable questions and explain every refusal."""
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    refused = [r for r in ans if r["answer"] == REFUSAL]
    answered = [r for r in ans if r["answer"] != REFUSAL]
    return {"n": len(rows), "answerable": len(ans), "unanswerable": len(una),
            "correct_answerable": _mean([r["correct"] for r in ans]),
            "correct_unanswerable": _mean([r["correct"] for r in una]),
            "refused_answerable": len(refused),
            "refused_by_router": sum(r.get("route") == "out_of_scope" for r in refused),
            "refused_after_grading": sum(r.get("route") != "out_of_scope" for r in refused),
            "correct_when_answered": _mean([r["correct"] for r in answered])}
