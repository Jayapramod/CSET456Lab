from pathlib import Path
from collections import Counter

REPO_PATH = Path("data/flask")

# Extensions that we consider source code
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


def analyze_repository():

    all_files = []
    source_files = []

    language_counter = Counter()
    extension_counter = Counter()

    total_loc = 0

    for path in REPO_PATH.rglob("*"):

        # Skip .git directory
        if ".git" in path.parts:
            continue

        if path.is_file():

            all_files.append(path)

            extension = path.suffix.lower()

            extension_counter[extension or "[no extension]"] += 1

            if extension in SOURCE_EXTENSIONS:

                source_files.append(path)

                language = SOURCE_EXTENSIONS[extension]
                language_counter[language] += 1

                total_loc += count_lines(path)

    directories = [
        p for p in REPO_PATH.rglob("*")
        if p.is_dir() and ".git" not in p.parts
    ]

    print("\n========== FLASK REPOSITORY INVENTORY ==========\n")

    print(f"Repository name       : Flask")
    print(f"Total files           : {len(all_files)}")
    print(f"Source-code files     : {len(source_files)}")
    print(f"Directories           : {len(directories)}")
    print(f"Total source LOC      : {total_loc}")

    print("\nProgramming Languages:")

    for language, count in language_counter.most_common():
        print(f"  {language}: {count} files")

    print("\nFile Type Distribution:")

    for extension, count in extension_counter.most_common():
        print(f"  {extension}: {count}")

    # Largest source files
    largest_files = []

    for file_path in source_files:
        loc = count_lines(file_path)
        largest_files.append((loc, file_path))

    largest_files.sort(reverse=True)

    print("\nLargest Source Files:")

    for loc, path in largest_files[:10]:
        print(f"  {loc:6} LOC  {path}")

    print("\nLOC per Source File:")

    average_loc = (
        total_loc / len(source_files)
        if source_files
        else 0
    )

    print(f"Average LOC per source file: {average_loc:.2f}")


if __name__ == "__main__":
    analyze_repository()