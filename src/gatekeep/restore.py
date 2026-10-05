"""Bring a backup (data/, cache/, results/) back into a fresh Kaggle session.

Kaggle unpacks a .zip uploaded to a dataset into plain folders, so a backup shows up either as a gatekeep_backup*.zip
file or as <something>/cache/llm.sqlite next to <something>/data. Both are handled; the fullest backup wins.
"""
import glob, os, shutil, zipfile

PARTS = ("data", "cache", "results", "models")


def restore(input_dir="/kaggle/input", dest="."):
    """Returns a description of what was restored, or None when no backup is attached."""
    folders = [os.path.dirname(os.path.dirname(p)) for p in glob.glob(f"{input_dir}/**/cache/llm.sqlite", recursive=True)]
    if folders:
        src = max(folders, key=lambda f: os.path.getsize(os.path.join(f, "cache", "llm.sqlite")))  # most cached replies
        for part in PARTS:
            if os.path.isdir(os.path.join(src, part)):
                shutil.copytree(os.path.join(src, part), os.path.join(dest, part), dirs_exist_ok=True)
        # The fine-tuned weights (GBs) usually live only in a notebook's output. The small backup carries models/ckpts.json
        # WITHOUT the weights, so look for the weights themselves, not for that file.
        if not glob.glob(os.path.join(src, "models", "*", "model.safetensors")):
            holders = [os.path.dirname(os.path.dirname(os.path.dirname(p)))
                       for p in glob.glob(f"{input_dir}/**/models/*/model.safetensors", recursive=True)]
            if holders:
                shutil.copytree(os.path.join(holders[0], "models"), os.path.join(dest, "models"), dirs_exist_ok=True)
        return f"folders in {src}"
    zips = glob.glob(f"{input_dir}/**/gatekeep_backup*.zip", recursive=True)
    if zips:
        newest = max(zips, key=os.path.getmtime)
        zipfile.ZipFile(newest).extractall(dest)
        return f"zip {newest}"
    return None
