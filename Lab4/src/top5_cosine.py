import pandas as pd
import sys


# -----------------------------------
# Check command-line argument
# -----------------------------------

if len(sys.argv) != 2:
    print("Usage:")
    print("python src/top5_cosine.py <csv_path>")
    sys.exit(1)


# -----------------------------------
# Get CSV path
# -----------------------------------

file_path = sys.argv[1]

print(f"\nLoading: {file_path}")


# -----------------------------------
# Read CSV
# -----------------------------------

df = pd.read_csv(file_path)


# -----------------------------------
# Check required columns
# -----------------------------------

required_columns = [
    "token_1",
    "token_2",
    "cosine_similarity"
]

for column in required_columns:

    if column not in df.columns:
        print(f"ERROR: Missing column '{column}'")
        sys.exit(1)


# -----------------------------------
# Convert similarity to numeric
# -----------------------------------

df["cosine_similarity"] = pd.to_numeric(
    df["cosine_similarity"],
    errors="coerce"
)

df = df.dropna(
    subset=["cosine_similarity"]
)


# -----------------------------------
# Sort highest → lowest
# -----------------------------------

df = df.sort_values(
    by="cosine_similarity",
    ascending=False
)


# -----------------------------------
# Get Top 5
# -----------------------------------

top_5 = df.head(5)


# -----------------------------------
# Display
# -----------------------------------

print("\n==========================================")
print("TOP 5 PAIRWISE COSINE SIMILARITY")
print("==========================================")

for i, row in top_5.reset_index(drop=True).iterrows():

    print(
        f"{i + 1}. "
        f"{row['token_1']} <-> {row['token_2']} "
        f"= {row['cosine_similarity']:.6f}"
    )


print("\n==========================================")
print(f"Total pairs: {len(df)}")
print("==========================================")