"""Build the two v2 notebooks. They clone the repository's default branch.
10_finetune_v2.ipynb, from notebook 04: same setup and the same official training script, but the training files come
from the private gatekeep-gates-v2 dataset and nothing is evaluated here.
11_eval_v2.ipynb, from notebook 08: same setup (clone, pip, GPU, secret, restore, weights check), then plan Tasks 9-11
as CLI calls: calibrate, per-gate results, V1/V2/V3, hand-written questions, report, OOD, backup."""
import json


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(True)}


def write(nb, path):
    json.dump(nb, open(path, "w", encoding="utf-8"), indent=1)
    print("wrote", path, "with", len(nb["cells"]), "cells")


# --- notebook 10 ---
nb04 = json.load(open("notebooks/04_finetune_laya.ipynb", encoding="utf-8"))
cells = nb04["cells"]
fetch = code('''import glob, os, shutil, sys
sys.path.insert(0, "src")   # the training cell imports gatekeep.data.gold_for
os.makedirs("data/gates", exist_ok=True)
for gate in ("route", "grade", "grounded", "sufficient"):
    found = glob.glob(f"/kaggle/input/**/{gate}_train.jsonl", recursive=True)
    assert found, f"{gate}_train.jsonl not found: attach the private gatekeep-gates-v2 dataset"
    shutil.copy(found[0], f"data/gates/{gate}_train.jsonl")
print(sorted(os.listdir("data/gates")))''')
write({**nb04, "cells": [
    md("# GateKeep · notebook 10 · fine-tune the v2 gates (training only)\n\n"
       "**Attach:** the private dataset `gatekeep-gates-v2`. **Settings:** GPU T4 ×2, Internet On. No API key needed.\n"
       "Run with **Save Version → Save & Run All (Commit)**, otherwise the weights are lost. Then download the output "
       "(`GateKeep/models/`) and follow Task 9 of the v2 plan."),
    cells[2], cells[3], cells[4],          # clone, pip install, nvidia-smi
    md("## training files"), fetch,
    cells[11], cells[12],                  # the official training script, unchanged
    cells[13], cells[14],                  # build items with gold_for (now soft targets) and train each gate
    code("!du -sh models/* && cat models/ckpts.json"),
]}, "notebooks/10_finetune_v2.ipynb")


# --- notebook 11 ---
nb08 = json.load(open("notebooks/08_ood_sklearn.ipynb", encoding="utf-8"))


def find(needle):
    for c in nb08["cells"]:
        if c["cell_type"] == "code" and needle in "".join(c["source"]):
            return dict(c)
    raise SystemExit(f"no notebook 08 cell contains {needle!r}")


clone = find('git", "clone"')
pip, smi, secret = find("pip install"), find("nvidia-smi"), find("UserSecretsClient")
restore, weights = find("from gatekeep.restore import restore"), find("fine-tuned weights missing")
weights["source"] = "".join(weights["source"]).replace("notebook 04", "notebook 10").splitlines(True)
write({**nb08, "cells": [
    md("# GateKeep · notebook 11 · v2 evaluation (plan Tasks 9-11)\n\n"
       "**Attach:** `gatekeep_data` (the new version) and notebook 10's output. **Settings:** GPU T4, Internet On, "
       "secret `OLLAMA_API_KEY` (the new key).\n"
       "Run with **Save Version → Save & Run All**, then download `gatekeep_backup.zip` from Output."),
    md("## 1 · code, packages, key"), clone, pip, smi, secret,
    md("## 2 · restore (cache and data from the backup, models from notebook 10's output)"), restore, weights,
    md("## 3 · calibrate on the dev split"), code("!python -m gatekeep.cli calibrate"),
    md("## 4 · per-gate results"),
    code("!python -m gatekeep.cli evalgates llm,sklearn,laya-zero,laya-ft\n!cat results/tau.json"),
    md("## 5 · V1, V2, V3 on the test set"), code("!python -m gatekeep.cli variants V1,V2,V3"),
    md("## 6 · the 80 hand-written questions"),
    code('!python -m gatekeep.cli variants V1,V3 "" results/tau.json 0 hw_ data/handwritten.jsonl'),
    md("## 7 · breakdown, verdict and plots"),
    code("!python -m gatekeep.cli breakdown\n!python -m gatekeep.cli report"),
    md("## 8 · scikit-learn docs (out of distribution)"),
    code('''if not os.path.isdir("data_book"): shutil.move("data", "data_book"); os.mkdir("data")
try:
    for cmd in (["python", "scripts/fetch_sklearn_docs.py"],
                ["python", "-m", "gatekeep.cli", "prepare_dir", "data/sk"],
                ["python", "-m", "gatekeep.cli", "questions", "0", "0", "200"],
                ["python", "-m", "gatekeep.cli", "variants", "V1,V3", "", "results/tau.json", "0", "ood_"]):
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(r.stdout[-3000:])
        assert r.returncode == 0, cmd
finally:
    shutil.rmtree("data"); shutil.move("data_book", "data")   # always put the book's data back'''),
    md("## 9 · back up"),
    code("!cd /kaggle/working/GateKeep && zip -rq /kaggle/working/gatekeep_backup.zip data cache results "
         "models/ckpts.json models/*/calibration.json && ls -lh /kaggle/working/gatekeep_backup.zip"),
]}, "notebooks/11_eval_v2.ipynb")
