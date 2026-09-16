from git import Repo
from pathlib import Path


REPOSITORIES = {
    "flask": "https://github.com/pallets/flask.git",
    "requests": "https://github.com/psf/requests.git",
    "pytest": "https://github.com/pytest-dev/pytest.git",
    "fastapi": "https://github.com/fastapi/fastapi.git",
    "scikit-learn": "https://github.com/scikit-learn/scikit-learn.git"
}


BASE_PATH = Path("data/repositories")


def clone_repositories():

    BASE_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    for name, url in REPOSITORIES.items():

        repo_path = BASE_PATH / name

        if repo_path.exists():
            print(f"[SKIP] {name} already exists")
            continue

        print(f"[CLONE] {name}")

        Repo.clone_from(
            url,
            repo_path
        )

        print(f"[DONE] {name}")


if __name__ == "__main__":
    clone_repositories()

