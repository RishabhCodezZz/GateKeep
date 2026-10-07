"""Build notebooks/11_eval_v2.ipynb from notebook 08: same setup (clone, pip, GPU, secret, restore, weights check),
then plan Tasks 9-11 as CLI calls: calibrate, per-gate results, V1/V2/V3, hand-written questions, report, OOD, backup."""
import json

src = json.load(open("notebooks/08_ood_sklearn.ipynb", encoding="utf-8"))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(True)}


def find(needle):
    for c in src["cells"]:
        if c["cell_type"] == "code" and needle in "".join(c["source"]):
            return dict(c)
    raise SystemExit(f"no notebook 08 cell contains {needle!r}")


clone = find('git", "clone"')
pip, smi, secret = find("pip install"), find("nvidia-smi"), find("UserSecretsClient")
restore, weights = find("from gatekeep.restore import restore"), find("fine-tuned weights missing")

OLD = '["git", "clone", "https://github.com/RishabhCodezZz/GateKeep.git"]'
NEW = '["git", "clone", "-b", "v2", "https://github.com/RishabhCodezZz/GateKeep.git"]'
joined = "".join(clone["source"])
assert OLD in joined, "clone command not found in notebook 08"
clone["source"] = joined.replace(OLD, NEW).splitlines(True)

ood = code('''if not os.path.isdir("data_book"): shutil.move("data", "data_book"); os.mkdir("data")
try:
    for cmd in (["python", "scripts/fetch_sklearn_docs.py"],
                ["python", "-m", "gatekeep.cli", "prepare_dir", "data/sk"],
                ["python", "-m", "gatekeep.cli", "questions", "0", "0", "200"],
                ["python", "-m", "gatekeep.cli", "variants", "V1,V3", "", "results/tau.json", "0", "ood_"]):
        print(subprocess.run(cmd, capture_output=True, text=True).stdout[-3000:])
finally:
    shutil.rmtree("data"); shutil.move("data_book", "data")   # always put the book's data back''')

out = dict(src)
out["cells"] = [
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
    md("## 8 · scikit-learn docs (out of distribution)"), ood,
    md("## 9 · back up"),
    code("!cd /kaggle/working/GateKeep && zip -rq /kaggle/working/gatekeep_backup.zip data cache results "
         "models/ckpts.json models/*/calibration.json && ls -lh /kaggle/working/gatekeep_backup.zip"),
]
json.dump(out, open("notebooks/11_eval_v2.ipynb", "w", encoding="utf-8"), indent=1)
print("wrote notebooks/11_eval_v2.ipynb with", len(out["cells"]), "cells")
