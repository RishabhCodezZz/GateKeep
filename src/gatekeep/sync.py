"""Persist private data between Kaggle sessions via a private Hugging Face dataset repo."""
import os

from huggingface_hub import HfApi, snapshot_download


def push(repo, folders=("data", "cache", "results", "models")):
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    for f in folders:
        if os.path.isdir(f):
            api.upload_folder(folder_path=f, path_in_repo=f, repo_id=repo, repo_type="dataset")


def pull(repo):
    try:
        snapshot_download(repo, repo_type="dataset", local_dir=".", token=os.environ["HF_TOKEN"])
    except Exception as e:  # first run: repo does not exist yet
        print("sync.pull skipped:", repr(e)[:120])
