# Handoff: GateKeep v2, Kaggle fine-tuning

Date: 2026-10-07. Branch: `v2` (pushed to GitHub as `origin/v2`; `main` is untouched).

## Where things stand

Tasks 1 to 7 of `docs/superpowers/plans/2026-10-07-gatekeep-v2.md` are done and reviewed, and Task 8's code is pushed. All 111 tests pass in `.venv`.

- Router v2 is in place. Its classes are `retrieve`, `direct` and `off_topic`, and the book's coverage is now decided after retrieval.
- Training uses soft targets with no copied rows. The evidence window is 2,000 characters, which fits Laya's input: 0% of rows go over the 768-token budget.
- Calibration is refit on the dev split, with histogram binning that now reaches the yes/no gates. Every cascade threshold aims for at least 0.90 accuracy.
- The v2 success criteria are written in the spec before any v2 result exists.
- The new gate data is built. The route gate has 972 train rows (804 retrieve, 96 off_topic, 72 direct), 192 dev rows and 453 test rows.
- `gates_v2.zip` is in the repo folder. It holds the four training files, is 1.2 MB, is private, and is git-ignored.
- The v1 results are kept in `results/v1/`.

## What you do now

### 1. Upload the training files to Kaggle

1. Go to kaggle.com, then **Datasets** and **New Dataset**.
2. Upload `C:\Users\risha\OneDrive\Desktop\GateKeep\gates_v2.zip`.
3. Name it `gatekeep-gates-v2` and set visibility to **Private**. This matters, because the files come from the book. Then click Create.
4. Once it shows on Kaggle, you can delete the local copy (optional; git ignores it anyway).

### 2. Run notebook 10

1. In Kaggle, go to **Code** and **New Notebook**. Then use **File** and **Import Notebook**, and pick `notebooks\10_finetune_v2.ipynb` from the repo.
2. Under **Input** and **Add Input**, attach your private dataset `gatekeep-gates-v2`. You don't need any other inputs or the API key.
3. In **Settings**, set the accelerator to **GPU T4 x2** and turn Internet **On**.
4. Click **Save Version** and choose **Save & Run All (Commit)**. An interactive run throws the weights away when the session ends.
5. Wait about 25 to 40 minutes. For each of the four gates the log should show:
   - `N training sequences (0 dropped)` (route should be 972)
   - four epochs
   - `Fitted calibration temperatures ...`
   - `Model successfully saved ...`

   The first cell should print commit `9c03530` on branch v2. If it shows something else, stop and tell me.

### 3. Download the weights

1. When the version is finished, open the notebook's **Output** tab and download the output. You get one zip of about 3.4 GB.
2. You only need the `models` folder inside `GateKeep`:
   ```
   GateKeep/models/ckpts.json
   GateKeep/models/route/
   GateKeep/models/grade/
   GateKeep/models/grounded/
   GateKeep/models/sufficient/
   ```
3. Open the zip in Explorer and copy that whole `models` folder into `C:\Users\risha\OneDrive\Desktop\GateKeep\`, so that `GateKeep\models\ckpts.json` exists. Git ignores `models/`.

Then tell me "weights are in". I'll check them and carry on with Tasks 9 to 12 on your PC: calibration, per-gate results, V1 to V3, the hand-written set, the scikit-learn run, and the README.

## If something goes wrong

- **The notebook stops at the training-files cell:** the dataset isn't attached, or it has a different name. Attach `gatekeep-gates-v2` and run again.
- **Out of GPU memory:** check that the accelerator is **T4 x2** and not P100 or a single T4.
- **The route log doesn't show 972 training sequences:** the notebook cloned the wrong branch. Send me the first cell's output.
- **Download is slow or fails:** you can download each gate folder from the Output tab one at a time instead of all at once.

## For the next session (if this chat is reset)

- Progress ledger: `.superpowers/sdd/2026-10-07-gatekeep-v2/progress.md` (git-ignored; trust it and `git log main..v2`).
- Resume at Task 9 of the plan once `models/` is in place.
- The Ollama key now lives in your Windows user environment. Commands load it with `$env:OLLAMA_API_KEY = [Environment]::GetEnvironmentVariable('OLLAMA_API_KEY','User')`.
- Ollama's usage endpoint no longer reports the monthly share, so check your quota on the ollama.com dashboard.
- There are 5 small simplifications from the ponytail review, and they're logged in the ledger for the final cleanup.
