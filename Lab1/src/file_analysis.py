from pathlib import Path
import csv

REPO_PATH = Path("data/flask")
OUTPUT_PATH = Path("output/file_dataset.csv")

SOURCE_EXTENSIONS = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".java": "Java",
    ".c": "C",
    ".cpp": "C++",
    ".h": "C/C++ Header",
    ".hpp": "C++ Header",
    ".go": "Go",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".php": "PHP",
    ".sh": "Shell",
}


def count_lines(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def create_dataset():

    rows = []

    for path in REPO_PATH.rglob("*"):

        if ".git" in path.parts:
            continue

        if not path.is_file():
            continue

        extension = path.suffix.lower()

        # Only source files
        if extension not in SOURCE_EXTENSIONS:
            continue

        language = SOURCE_EXTENSIONS[extension]

        loc = count_lines(path)

        size_bytes = path.stat().st_size

        relative_path = path.relative_to(REPO_PATH)

        rows.append({
            "file_path": str(relative_path),
            "language": language,
            "extension": extension,
            "loc": loc,
            "size_bytes": size_bytes
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        fieldnames = [
            "file_path",
            "language",
            "extension",
            "loc",
            "size_bytes"
        ]

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {OUTPUT_PATH}")
    print(f"Number of source files: {len(rows)}")


if __name__ == "__main__":
    create_dataset()
