from git import Repo
from pathlib import Path

REPO_URL = "https://github.com/pallets/flask.git"
REPO_PATH = Path("data/flask")


def clone_repository():
    if REPO_PATH.exists():
        print("Repository already exists.")
        return

    print("Cloning Flask repository...")
    Repo.clone_from(REPO_URL, REPO_PATH)
    print("Repository cloned successfully.")


if __name__ == "__main__":
    clone_repository()
