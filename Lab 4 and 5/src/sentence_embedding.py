import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


# -----------------------------------
# Paths
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "20_tokens.csv"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------------
# Load CSV
# -----------------------------------

# keep_default_na=False is important.
# It prevents "None" from becoming NaN.
df = pd.read_csv(
    INPUT_FILE,
    header=None,
    names=["rank", "token", "count"],
    keep_default_na=False
)


# -----------------------------------
# Extract tokens
# -----------------------------------

tokens = df["token"].astype(str).tolist()

print("Tokens:")

for i, token in enumerate(tokens, start=1):
    print(f"{i}. {token}")


# -----------------------------------
# Load Sentence-BERT
# -----------------------------------

print("\nLoading Sentence-BERT model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)


# -----------------------------------
# Generate embeddings
# -----------------------------------

print("\nGenerating embeddings...")

embeddings = model.encode(
    tokens,
    convert_to_numpy=True,
    show_progress_bar=True
)


# -----------------------------------
# Check dimensions
# -----------------------------------

print("\nEmbedding shape:")
print(embeddings.shape)


# -----------------------------------
# Create DataFrame
# -----------------------------------

embedding_df = pd.DataFrame(embeddings)

embedding_df.insert(
    0,
    "token",
    tokens
)


# -----------------------------------
# Save embeddings
# -----------------------------------

OUTPUT_FILE = OUTPUT_DIR / "sentence_embeddings.csv"

embedding_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# -----------------------------------
# Display result
# -----------------------------------

print("\n======================================")
print("SENTENCE EMBEDDINGS")
print("======================================")

print(f"Number of tokens: {len(tokens)}")
print(f"Embedding dimensions: {embeddings.shape[1]}")

print("\nFirst few embeddings:")
print(embedding_df.head())

print(f"\nSaved to: {OUTPUT_FILE}")