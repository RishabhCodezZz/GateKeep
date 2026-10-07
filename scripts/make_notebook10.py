"""Build notebooks/10_finetune_v2.ipynb from notebook 04: same setup and the same official training script,
but the training files come from the private gatekeep-gates-v2 dataset and nothing is evaluated here."""
import json

src = json.load(open("notebooks/04_finetune_laya.ipynb", encoding="utf-8"))
cells = src["cells"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(True)}


# the v2 code lives on branch v2, so clone that branch (the cell's git-pull branch for an existing folder is left alone)
OLD = '["git", "clone", "https://github.com/RishabhCodezZz/GateKeep.git"]'
NEW = '["git", "clone", "-b", "v2", "https://github.com/RishabhCodezZz/GateKeep.git"]'
clone = dict(cells[2])
joined = "".join(clone["source"])
if OLD not in joined:
    raise SystemExit("clone command not found in notebook 04 cell 2")
clone["source"] = joined.replace(OLD, NEW).splitlines(True)

fetch = code('''import glob, os, shutil, sys
sys.path.insert(0, "src")   # the training cell imports gatekeep.data.gold_for
os.makedirs("data/gates", exist_ok=True)
for gate in ("route", "grade", "grounded", "sufficient"):
    found = glob.glob(f"/kaggle/input/**/{gate}_train.jsonl", recursive=True)
    assert found, f"{gate}_train.jsonl not found: attach the private gatekeep-gates-v2 dataset"
    shutil.copy(found[0], f"data/gates/{gate}_train.jsonl")
print(sorted(os.listdir("data/gates")))''')

out = dict(src)
out["cells"] = [
    md("# GateKeep · notebook 10 · fine-tune the v2 gates (training only)\n\n"
       "**Attach:** the private dataset `gatekeep-gates-v2`. **Settings:** GPU T4 ×2, Internet On. No API key needed.\n"
       "Run with **Save Version → Save & Run All (Commit)**, otherwise the weights are lost. Then download the output "
       "(`GateKeep/models/`) and follow Task 9 of the v2 plan."),
    clone, cells[3], cells[4],             # clone (branch v2), pip install, nvidia-smi
    md("## training files"), fetch,
    cells[11], cells[12],                  # the official training script, unchanged
    cells[13], cells[14],                  # build items with gold_for (now soft targets) and train each gate
    code("!du -sh models/* && cat models/ckpts.json"),
]
json.dump(out, open("notebooks/10_finetune_v2.ipynb", "w", encoding="utf-8"), indent=1)
print("wrote notebooks/10_finetune_v2.ipynb with", len(out["cells"]), "cells")
