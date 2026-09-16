import os
import pandas as pd
from git import Repo


# ============================================================
# CHANGE THIS TO THE REPOSITORY YOU WANT TO PROCESS
# ============================================================

REPO_NAME = "flask"


# ============================================================
# PATHS
# ============================================================

REPO_PATH = os.path.join("data", "repos", REPO_NAME)
SOURCE_FILE = os.path.join(
    "data",
    "output",
    f"{REPO_NAME}_source_code.csv"
)

COMMIT_FILE = os.path.join(
    "data",
    "output",
    f"{REPO_NAME}_commit_history.csv"
)

OUTPUT_FILE = os.path.join(
    "data",
    "output",
    f"{REPO_NAME}_combined.csv"
)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(SOURCE_FILE):
    print(f"[ERROR] Source dataset not found: {SOURCE_FILE}")
    exit(1)

if not os.path.exists(COMMIT_FILE):
    print(f"[ERROR] Commit dataset not found: {COMMIT_FILE}")
    exit(1)

if not os.path.exists(REPO_PATH):
    print(f"[ERROR] Repository not found: {REPO_PATH}")
    exit(1)


# ============================================================
# LOAD EXISTING DATASETS
# ============================================================

source_df = pd.read_csv(SOURCE_FILE)
commit_df = pd.read_csv(COMMIT_FILE)

print(f"Source records: {len(source_df)}")
print(f"Commit records: {len(commit_df)}")


# ============================================================
# SELECT RELEVANT SOURCE-CODE ATTRIBUTES
# ============================================================

source_df = source_df[
    [
        "repository",
        "file_name",
        "file_path",
        "file_extension",
        "line_count",
        "character_count",
        "source_code"
    ]
]


# ============================================================
# LOAD GIT REPOSITORY
# ============================================================

repo = Repo(REPO_PATH)


# ============================================================
# EXTRACT FILE-LEVEL COMMIT HISTORY
# ============================================================

file_commit_data = []


for commit in repo.iter_commits():

    commit_hash = commit.hexsha

    author_name = commit.author.name

    commit_date = commit.committed_datetime

    commit_message = commit.message.strip()


    # Get files changed by this commit
    try:

        parent = commit.parents[0]

        diffs = parent.diff(commit)

        for diff in diffs:

            # Determine file path
            file_path = diff.b_path or diff.a_path

            if not file_path:
                continue


            # Normalize path separators
            file_path = file_path.replace("\\", "/")


            # Get additions and deletions
            try:
                additions = diff.diff.decode(
                    "utf-8",
                    errors="ignore"
                ).count("\n")
            except Exception:
                additions = 0


            # Use Git stats where possible
            try:

                stats = commit.stats.files

                if file_path in stats:

                    additions = stats[file_path]["insertions"]
                    deletions = stats[file_path]["deletions"]

                else:

                    additions = 0
                    deletions = 0

            except Exception:

                additions = 0
                deletions = 0


            lines_changed = additions + deletions


            file_commit_data.append({

                "repository": REPO_NAME,

                "file_path": file_path,

                "commit_hash": commit_hash,

                "author_name": author_name,

                "commit_date": commit_date,

                "commit_message": commit_message,

                "additions": additions,

                "deletions": deletions,

                "lines_changed": lines_changed

            })


    except Exception as e:

        print(
            f"[WARNING] Could not process commit "
            f"{commit_hash}: {e}"
        )


# ============================================================
# CREATE FILE-LEVEL COMMIT DATAFRAME
# ============================================================

file_commit_df = pd.DataFrame(file_commit_data)


print(
    f"File-level commit records: "
    f"{len(file_commit_df)}"
)


# ============================================================
# MERGE SOURCE CODE + FILE COMMIT HISTORY
# ============================================================

combined_df = pd.merge(
    source_df,
    file_commit_df,
    on=["repository", "file_path"],
    how="inner"
)


# ============================================================
# REMOVE UNNECESSARY COLUMNS
# ============================================================

# author_email is intentionally not included.
#
# files_changed is also not included because it describes
# the whole commit rather than the individual file.


# ============================================================
# REMOVE DUPLICATES
# ============================================================

combined_df = combined_df.drop_duplicates()


# ============================================================
# SAVE DATASET
# ============================================================

combined_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("COMBINED DATASET CREATED")
print("=" * 60)

print(f"Repository : {REPO_NAME}")
print(f"Rows       : {len(combined_df)}")
print(f"Columns    : {len(combined_df.columns)}")
print(f"Output     : {OUTPUT_FILE}")

print("\nColumns:")
for column in combined_df.columns:
    print(f" - {column}")

print("=" * 60)
