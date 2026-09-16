import os
import pandas as pd


# ============================================================
# DIRECTORIES
# ============================================================

OUTPUT_DIR = "output"
COMBINED_DIR = os.path.join(OUTPUT_DIR, "combined")


# Create combined directory
os.makedirs(COMBINED_DIR, exist_ok=True)


# ============================================================
# FIND ALL SOURCE CODE DATASETS
# ============================================================

source_files = [
    file
    for file in os.listdir(OUTPUT_DIR)
    if file.endswith("_source_code.csv")
]


if not source_files:
    print("[ERROR] No source-code CSV files found.")
    exit(1)


print("=" * 60)
print("COMBINING REPOSITORY DATASETS")
print("=" * 60)

print(f"Source datasets found: {len(source_files)}")


# Store all combined datasets
all_combined_data = []


# ============================================================
# PROCESS EACH REPOSITORY
# ============================================================

for source_file in source_files:

    # --------------------------------------------------------
    # Get repository name
    # --------------------------------------------------------

    repo_name = source_file.replace(
        "_source_code.csv",
        ""
    )

    print("\n" + "-" * 60)
    print(f"Processing: {repo_name}")
    print("-" * 60)


    # --------------------------------------------------------
    # File paths
    # --------------------------------------------------------

    source_path = os.path.join(
        OUTPUT_DIR,
        source_file
    )

    commit_file = f"{repo_name}_commit_history.csv"

    commit_path = os.path.join(
        OUTPUT_DIR,
        commit_file
    )


    # --------------------------------------------------------
    # Check matching commit dataset
    # --------------------------------------------------------

    if not os.path.exists(commit_path):

        print(
            f"[WARNING] Commit dataset not found: "
            f"{commit_file}"
        )

        continue


    # --------------------------------------------------------
    # Read CSV files
    # --------------------------------------------------------

    source_df = pd.read_csv(source_path)

    commit_df = pd.read_csv(commit_path)


    print(f"Source records : {len(source_df)}")
    print(f"Commit records : {len(commit_df)}")


    # ========================================================
    # SELECT RELEVANT SOURCE CODE FEATURES
    # ========================================================

    source_columns = [
        "repository",
        "file_name",
        "file_path",
        "file_extension",
        "line_count",
        "character_count",
        "source_code"
    ]

    # Keep only columns that actually exist
    source_columns = [
        column
        for column in source_columns
        if column in source_df.columns
    ]

    source_df = source_df[source_columns]


    # ========================================================
    # SELECT RELEVANT COMMIT FEATURES
    # ========================================================

    # We don't include:
    #
    # author_email
    # commit_hash
    # commit_message
    # commit_date
    #
    # because we are aggregating commit history at the
    # repository level.
    #
    # Instead, we calculate useful repository statistics.

    commit_count = len(commit_df)


    total_files_changed = (
        commit_df["files_changed"].sum()
        if "files_changed" in commit_df.columns
        else 0
    )


    total_insertions = (
        commit_df["insertions"].sum()
        if "insertions" in commit_df.columns
        else 0
    )


    total_deletions = (
        commit_df["deletions"].sum()
        if "deletions" in commit_df.columns
        else 0
    )


    total_lines_changed = (
        commit_df["lines_changed"].sum()
        if "lines_changed" in commit_df.columns
        else 0
    )


    # Number of unique contributors
    unique_authors = (
        commit_df["author_name"].nunique()
        if "author_name" in commit_df.columns
        else 0
    )


    # ========================================================
    # ADD REPOSITORY-LEVEL COMMIT FEATURES
    # ========================================================

    source_df["commit_count"] = commit_count

    source_df["total_files_changed"] = total_files_changed

    source_df["total_insertions"] = total_insertions

    source_df["total_deletions"] = total_deletions

    source_df["total_lines_changed"] = total_lines_changed

    source_df["unique_authors"] = unique_authors


    # ========================================================
    # SAVE COMBINED DATASET FOR THIS REPOSITORY
    # ========================================================

    output_file = os.path.join(
        COMBINED_DIR,
        f"{repo_name}_combined.csv"
    )


    source_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )


    print(
        f"[SUCCESS] Created: {output_file}"
    )


    print(
        f"Combined records: {len(source_df)}"
    )


    # Add to master dataset
    all_combined_data.append(source_df)


# ============================================================
# CREATE ONE MASTER DATASET
# ============================================================

if all_combined_data:

    master_df = pd.concat(
        all_combined_data,
        ignore_index=True
    )


    master_file = os.path.join(
        COMBINED_DIR,
        "all_repositories_combined.csv"
    )


    master_df.to_csv(
        master_file,
        index=False,
        encoding="utf-8"
    )


    print("\n" + "=" * 60)
    print("MASTER DATASET CREATED")
    print("=" * 60)

    print(
        f"Total records: {len(master_df)}"
    )

    print(
        f"Total columns: {len(master_df.columns)}"
    )

    print(
        f"Output: {master_file}"
    )


print("\n" + "=" * 60)
print("COMBINATION COMPLETED")
print("=" * 60)