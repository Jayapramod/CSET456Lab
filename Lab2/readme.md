# Lab 2 – Repository Data Mining

This project focuses on mining useful information from five open-source GitHub repositories:

- Flask
- Requests
- Pytest
- FastAPI
- Scikit-learn

For each repository, two separate datasets were created: one containing **source code information** and another containing **commit history**.

## 1. Data Extraction

### Source Code Dataset

The source code of each repository was analyzed to extract relevant information such as:

- Repository name
- File name and file path
- File extension
- Number of lines
- Character count
- Source code

This resulted in one source-code CSV file for each repository.

### Commit History Dataset

The Git history of each repository was mined to collect information about its development activity, including:

- Commit hash
- Author
- Commit date
- Commit message
- Number of files changed
- Insertions
- Deletions
- Total lines changed

This resulted in one commit-history CSV file for each repository.

Overall, **10 CSV files** were generated from the five repositories.

## 2. Data Selection and Combination

The extracted attributes were reviewed to identify information that was relevant for further analysis. Unnecessary information, such as author email and other attributes that were not required for the final analysis, was removed.

The relevant information from the source-code and commit-history datasets was then combined for each repository.

The final datasets are stored separately in:

```text
data/output/combined/