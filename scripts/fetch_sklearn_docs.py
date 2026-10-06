"""Download the scikit-learn user guide pages (BSD-3-Clause, (c) The scikit-learn developers) into data/sk."""
import pathlib, urllib.error, urllib.request

PAGES = ["clustering", "svm", "tree", "ensemble", "linear_model", "model_evaluation", "cross_validation",
         "preprocessing", "feature_selection", "calibration", "naive_bayes", "neighbors", "decomposition",
         "grid_search", "impute", "outlier_detection", "manifold", "mixture", "neural_networks_supervised",
         "sgd", "compose", "metrics"]

out = pathlib.Path("data/sk")
out.mkdir(parents=True, exist_ok=True)
for p in PAGES:
    try:
        urllib.request.urlretrieve(f"https://scikit-learn.org/stable/modules/{p}.html", out / f"{p}.html")
    except urllib.error.HTTPError as e:
        print("skip", p, e.code)
print(len(list(out.glob("*.html"))), "pages in", out)
