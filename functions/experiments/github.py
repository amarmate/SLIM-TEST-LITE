from pathlib import Path
import os, subprocess, time


def init_or_update_repo():
    if not (DATA_DIR / ".git").exists():
        subprocess.run(["git", "clone", REPO_URL, str(DATA_DIR)], check=True)
    else:
        os.chdir(DATA_DIR)
        subprocess.run(["git", "fetch", "origin"], check=True)
        subprocess.run(["git", "reset", "--hard", "origin/main"], check=True)

def commit_and_push(filename: str, msg: str):
    os.chdir(DATA_DIR)
    subprocess.run(["git", "add", filename], check=True)
    subprocess.run(["git", "commit", "-m", msg], check=True)
    subprocess.run(["git", "push", "origin", "main"], check=True)
