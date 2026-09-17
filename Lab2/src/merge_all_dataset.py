import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

COMBINED_DIR = "output/combined"

OUTPUT_FILE = os.path.join(
    COMBINED_DIR,
    "all_repositories.csv"
)


# ============================================================
# FIND THE 5 COMBINED DATASETS
# ============================================================

csv_files = [
    file
    for file in os.listdir(COMBINED_DIR)
    if file.endswith("_combined.csv")
]


if not csv_files:
    print("[ERROR] No combined CSV files found.")
    exit(1)


print("=" * 60)
print("MERGING ALL REPOSITORY DATASETS")
print("=" * 60)


# ============================================================
# READ AND COMBINE DATASETS
# ============================================================

dataframes = []

for file in sorted(csv_files):

    file_path = os.path.join(
        COMBINED_DIR,
        file
    )

    print(f"Reading: {file}")

    df = pd.read_csv(file_path)

    print(f"  Rows: {len(df)}")

    dataframes.append(df)


# ============================================================
# MERGE ALL DATASETS
# ============================================================

final_df = pd.concat(
    dataframes,
    ignore_index=True
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

final_df = final_df.drop_duplicates()


# ============================================================
# SAVE FINAL DATASET
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET CREATED")
print("=" * 60)

print(f"Repositories : {len(dataframes)}")
print(f"Total rows   : {len(final_df)}")
print(f"Total columns: {len(final_df.columns)}")
print(f"Output       : {OUTPUT_FILE}")

print("\nRepositories included:")

if "repository" in final_df.columns:
    for repo in final_df["repository"].unique():
        count = len(
            final_df[final_df["repository"] == repo]
        )
        print(f"  {repo}: {count} rows")

print("=" * 60)