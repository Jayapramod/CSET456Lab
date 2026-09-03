from pydriller import Repository
from collections import Counter, defaultdict
from pathlib import Path
import csv

REPO_PATH = "data/flask"
OUTPUT_DIR = Path("output")


def mine_git_history():

    commit_count = 0

    contributors = Counter()

    changed_files = Counter()

    commits_per_month = Counter()

    files_per_month = defaultdict(list)

    additions = 0
    deletions = 0

    commit_rows = []

    for commit in Repository(REPO_PATH).traverse_commits():

        commit_count += 1

        author = commit.author.name
        contributors[author] += 1

        month = commit.author_date.strftime("%Y-%m")

        commits_per_month[month] += 1

        files_changed = len(commit.modified_files)

        files_per_month[month].append(files_changed)

        additions += commit.insertions
        deletions += commit.deletions

        for modified_file in commit.modified_files:

            if modified_file.old_path:
                changed_files[modified_file.old_path] += 1

            if modified_file.new_path:
                changed_files[modified_file.new_path] += 1

        commit_rows.append({
            "hash": commit.hash,
            "author": author,
            "date": commit.author_date.isoformat(),
            "message": commit.msg.split("\n")[0],
            "files_changed": files_changed,
            "insertions": commit.insertions,
            "deletions": commit.deletions
        })

    return {
        "commit_count": commit_count,
        "contributors": contributors,
        "changed_files": changed_files,
        "commits_per_month": commits_per_month,
        "files_per_month": files_per_month,
        "insertions": additions,
        "deletions": deletions,
        "commit_rows": commit_rows
    }
def save_commit_dataset(rows):

    output = OUTPUT_DIR / "commit_history.csv"

    with open(output, "w", newline="", encoding="utf-8") as f:

        fieldnames = [
            "hash",
            "author",
            "date",
            "message",
            "files_changed",
            "insertions",
            "deletions"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {output}")

def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    data = mine_git_history()

    print("\n========== GIT HISTORY ==========\n")

    print("Total commits:", data["commit_count"])

    print(
        "Total contributors:",
        len(data["contributors"])
    )

    print("\nMost active contributors:")

    for name, count in data["contributors"].most_common(10):
        print(f"{name}: {count} commits")

    print("\nMost frequently changed files:")

    for file, count in data["changed_files"].most_common(10):
        print(f"{file}: {count} changes")

    print("\nCommits per month:")

    for month, count in sorted(
        data["commits_per_month"].items()
    ):
        print(f"{month}: {count}")

    print("\nAverage files changed per month:")

    for month in sorted(data["files_per_month"]):

        values = data["files_per_month"][month]

        average = sum(values) / len(values)

        print(f"{month}: {average:.2f}")

    print("\nTotal insertions:", data["insertions"])
    print("Total deletions:", data["deletions"])

    save_commit_dataset(data["commit_rows"])


if __name__ == "__main__":
    main()
