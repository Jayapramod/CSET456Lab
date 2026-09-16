import os
import csv
from git import Repo


# ============================================================
# CHANGE THIS NAME TO MINE A DIFFERENT REPOSITORY
# ============================================================

REPO_NAME = "flask"


# ============================================================
# DIRECTORY CONFIGURATION
# ============================================================

REPO_PATH = os.path.join("data", REPO_NAME)
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CHECK REPOSITORY
# ============================================================

if not os.path.exists(REPO_PATH):
    print(f"[ERROR] Repository not found: {REPO_PATH}")
    print("Make sure the repository has been cloned first.")
    exit(1)


repo = Repo(REPO_PATH)


# ============================================================
# 1. MINE SOURCE CODE
# ============================================================

def mine_source_code():

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{REPO_NAME}_source_code.csv"
    )

    source_extensions = (
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".rb",
        ".php"
    )

    source_data = []

    for root, dirs, files in os.walk(REPO_PATH):

        # Don't scan Git internal files
        if ".git" in dirs:
            dirs.remove(".git")

        for filename in files:

            # Only consider source-code files
            if not filename.endswith(source_extensions):
                continue

            file_path = os.path.join(root, filename)

            relative_path = os.path.relpath(
                file_path,
                REPO_PATH
            )

            try:
                with open(
                    file_path,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as file:

                    content = file.read()

                line_count = len(content.splitlines())
                character_count = len(content)

                source_data.append({
                    "repository": REPO_NAME,
                    "file_name": filename,
                    "file_path": relative_path,
                    "file_extension": os.path.splitext(filename)[1],
                    "line_count": line_count,
                    "character_count": character_count,
                    "source_code": content
                })

            except Exception as e:

                print(
                    f"[WARNING] Could not read {relative_path}: {e}"
                )

    # Write CSV
    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        fieldnames = [
            "repository",
            "file_name",
            "file_path",
            "file_extension",
            "line_count",
            "character_count",
            "source_code"
        ]

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(source_data)

    print(
        f"[SUCCESS] Source code extracted: {output_file}"
    )

    print(
        f"[INFO] Source files found: {len(source_data)}"
    )


# ============================================================
# 2. MINE COMMIT HISTORY
# ============================================================

def mine_commit_history():

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{REPO_NAME}_commit_history.csv"
    )

    commit_data = []

    for commit in repo.iter_commits():

        commit_data.append({
            "repository": REPO_NAME,
            "commit_hash": commit.hexsha,
            "author_name": commit.author.name,
            "author_email": commit.author.email,
            "commit_date": commit.committed_datetime,
            "commit_message": commit.message.strip(),
            "files_changed": len(commit.stats.files),
            "insertions": commit.stats.total["insertions"],
            "deletions": commit.stats.total["deletions"],
            "lines_changed": (
                commit.stats.total["insertions"]
                + commit.stats.total["deletions"]
            )
        })

    # Write CSV
    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        fieldnames = [
            "repository",
            "commit_hash",
            "author_name",
            "author_email",
            "commit_date",
            "commit_message",
            "files_changed",
            "insertions",
            "deletions",
            "lines_changed"
        ]

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(commit_data)

    print(
        f"[SUCCESS] Commit history extracted: {output_file}"
    )

    print(
        f"[INFO] Commits found: {len(commit_data)}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print(f"Mining repository: {REPO_NAME}")
    print("=" * 60)

    mine_source_code()
    mine_commit_history()

    print("=" * 60)
    print("Mining completed!")
    print("=" * 60)
