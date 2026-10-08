"""Reading the results: where a variant wins or loses, in numbers, plus the verdict and plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gatekeep.graph import REFUSAL

ROUTER_REFUSALS = ("off_topic", "out_of_scope")  # v2 label, and v1's for the archived rows


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
            "refused_by_router": sum(r.get("route") in ROUTER_REFUSALS for r in refused),
            "refused_after_grading": sum(r.get("route") not in ROUTER_REFUSALS for r in refused),
            "correct_when_answered": _mean([r["correct"] for r in answered])}


def verdict(s_v1, s_v3, max_gap=0.025, min_ratio=5.0):
    gap = s_v1["correct"][0] - s_v3["correct"][0]
    ratio = s_v1["gate_llm_calls"][0] / max(s_v3["gate_llm_calls"][0], 1e-9)
    return {"quality_gap": gap, "call_ratio": ratio, "pass": gap <= max_gap and ratio >= min_ratio}


def plot_tradeoff(points, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, (calls, correct) in points.items():
        ax.scatter(calls, correct)
        ax.annotate(label, (calls, correct), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("LLM calls per question")
    ax.set_ylabel("answer correctness")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coverage(curves, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, pts in curves.items():
        ax.plot([c for _, c, _ in pts], [a for _, _, a in pts], marker="o", label=label)
    ax.set_xlabel("fraction of decisions Laya answers alone")
    ax.set_ylabel("accuracy on those decisions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
