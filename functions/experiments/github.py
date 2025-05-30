from pathlib import Path
import os, subprocess, time

def init_or_update_repo(config):
    os.system("git config user.name 'Mateus GP Bot'")
    os.system("git config user.email 'mbaptistaamaral@gmail.com'")

    prev_dir = os.getcwd()
    data_dir = Path("..") / config['DATA_DIR']
    os.chdir(data_dir)

    if not (data_dir / ".git").exists():
        subprocess.run(["git", "clone", config['REPO_URL'], str(data_dir)], check=True)
    else:
        subprocess.run(["git", "fetch", "origin"], check=True)
        subprocess.run(["git", "reset", "--hard", "origin/main"], check=True)

    os.chdir(prev_dir)
    
def commit_and_push(config: dict, filename: str, msg: str):
    prev_dir = os.getcwd()
    data_dir = Path("..") / config['DATA_DIR']
    os.chdir(data_dir)

    subprocess.run(["git", "add", filename], check=True)
    subprocess.run(["git", "commit", "-m", msg], check=True)
    subprocess.run(["git", "push", "origin", "main"], check=True)

    os.chdir(prev_dir)

def auto_commit_and_push(config: dict, msg: str):
    prev_dir = os.getcwd()
    data_dir = Path("..") / config['DATA_DIR']
    os.chdir(data_dir)

    subprocess.run(["git", "add", "."], check=True)
    result = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if result.returncode != 0:
        subprocess.run(["git", "commit", "-m", msg], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print(f"Changes committed and pushed with message: {msg}")
    else:
        print("Warning: No changes to commit! Skipping commit and push.")
    os.chdir(prev_dir)

