import ast
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CODE_DIR = BASE_DIR.parent / "Lab1" / "src"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# AST PATH EXTRACTION
# ============================================================

def get_node_type(node):
    return type(node).__name__


def extract_paths(node, path=None, paths=None):

    if paths is None:
        paths = []

    if path is None:
        path = []

    current_path = path + [get_node_type(node)]

    children = list(ast.iter_child_nodes(node))

    # Leaf node
    if not children:
        paths.append(current_path)
        return paths

    for child in children:

        extract_paths(
            child,
            current_path,
            paths
        )

    return paths


# ============================================================
# READ PYTHON FILES
# ============================================================

python_files = list(CODE_DIR.rglob("*.py"))

print(f"Found {len(python_files)} Python files.")


all_paths = []


for file in python_files:

    print(f"\nProcessing: {file}")

    try:

        source_code = file.read_text(
            encoding="utf-8"
        )

        tree = ast.parse(source_code)

        paths = extract_paths(tree)

        all_paths.extend(paths)

        print(
            f"Extracted {len(paths)} AST paths"
        )

    except Exception as e:

        print(
            f"Error processing {file}: {e}"
        )


# ============================================================
# BUILD VOCABULARY
# ============================================================

node_counter = Counter()

for path in all_paths:

    for node in path:

        node_counter[node] += 1


vocabulary = {
    node: index
    for index, node in enumerate(
        node_counter.keys()
    )
}


print("\n====================================")
print("AST VOCABULARY")
print("====================================")

print(
    f"Vocabulary size: {len(vocabulary)}"
)


# ============================================================
# CREATE PATH VECTORS
# ============================================================

embedding_size = 128

np.random.seed(42)

node_embeddings = np.random.normal(
    0,
    0.1,
    (
        len(vocabulary),
        embedding_size
    )
)


# ============================================================
# GENERATE CODE EMBEDDING
# ============================================================

def generate_code_embedding(paths):

    vectors = []

    for path in paths:

        path_vectors = []

        for node in path:

            index = vocabulary[node]

            path_vectors.append(
                node_embeddings[index]
            )

        if path_vectors:

            path_vector = np.mean(
                path_vectors,
                axis=0
            )

            vectors.append(path_vector)

    if not vectors:

        return np.zeros(embedding_size)

    # Aggregate all path vectors
    code_vector = np.mean(
        vectors,
        axis=0
    )

    return code_vector


# ============================================================
# GENERATE EMBEDDINGS FOR FILES
# ============================================================

results = []


for file in python_files:

    try:

        source_code = file.read_text(
            encoding="utf-8"
        )

        tree = ast.parse(source_code)

        paths = extract_paths(tree)

        embedding = generate_code_embedding(
            paths
        )

        row = {
            "file": str(file)
        }

        for i, value in enumerate(
            embedding
        ):

            row[f"dim_{i+1}"] = value

        results.append(row)

    except Exception as e:

        print(
            f"Error: {file}: {e}"
        )


# ============================================================
# SAVE
# ============================================================

embedding_df = pd.DataFrame(
    results
)


OUTPUT_FILE = (
    OUTPUT_DIR /
    "code2vec_embeddings.csv"
)


embedding_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n====================================")
print("CODE EMBEDDINGS GENERATED")
print("====================================")

print(
    f"Files embedded: {len(results)}"
)

print(
    f"Embedding dimensions: {embedding_size}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)
