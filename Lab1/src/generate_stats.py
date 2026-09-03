import json
import pandas as pd
from pathlib import Path


OUTPUT_DIR = Path("output")


def generate_statistics():

    # Read file-level dataset
    files_df = pd.read_csv(
        OUTPUT_DIR / "file_dataset.csv"
    )

    # Read commit dataset
    commits_df = pd.read_csv(
        OUTPUT_DIR / "commit_history.csv"
    )

    # Basic file statistics
    total_source_files = len(files_df)

    total_loc = int(files_df["loc"].sum())

    average_loc = (
        float(files_df["loc"].mean())
        if total_source_files > 0
        else 0
    )

    # Language statistics
    languages = (
        files_df["language"]
        .value_counts()
        .to_dict()
    )

    # File extension statistics
    file_types = (
        files_df["extension"]
        .value_counts()
        .to_dict()
    )

    # Largest source files
    largest_files = (
        files_df
        .sort_values("loc", ascending=False)
        .head(10)
        [["file_path", "loc", "size_bytes"]]
        .to_dict(orient="records")
    )

    # Git statistics
    total_commits = len(commits_df)

    contributors = (
        commits_df["author"]
        .nunique()
    )

    most_active = (
        commits_df["author"]
        .value_counts()
        .head(10)
        .to_dict()
    )

    total_insertions = int(
        commits_df["insertions"].sum()
    )

    total_deletions = int(
        commits_df["deletions"].sum()
    )

    # Most frequently changed files
    # This will be generated from the commit history
    # if frequently_changed_files.csv exists.
    frequent_files_path = (
        OUTPUT_DIR / "frequently_changed_files.csv"
    )

    if frequent_files_path.exists():

        frequent_df = pd.read_csv(
            frequent_files_path
        )

        frequently_changed = (
            frequent_df
            .head(10)
            .to_dict(orient="records")
        )

    else:
        frequently_changed = []

    # Final JSON structure
    statistics = {

        "repository": {
            "name": "Flask",
            "source": "https://github.com/pallets/flask"
        },

        "file_statistics": {
            "total_source_files": total_source_files,
            "total_loc": total_loc,
            "average_loc_per_source_file": round(
                average_loc, 2
            ),

            "languages": languages,

            "file_type_distribution": file_types,

            "largest_source_files": largest_files
        },

        "git_history": {

            "total_commits": total_commits,

            "total_contributors": contributors,

            "most_active_contributors": most_active,

            "total_insertions": total_insertions,

            "total_deletions": total_deletions,

            "frequently_changed_files":
                frequently_changed
        }
    }

    # Write JSON
    output_file = (
        OUTPUT_DIR / "repository_stats.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            statistics,
            f,
            indent=4
        )

    print(
        f"Created {output_file}"
    )


if __name__ == "__main__":
    generate_statistics()
