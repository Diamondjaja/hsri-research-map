"""
Step 2 (HSRI adaptation of 02_Embedding_and_UMAP.ipynb)
=========================================================

Builds a bilingual (English-first, Thai second) embedding text per project,
embeds with Cohere embed-multilingual-v3.0, reduces with UMAP (same params as
the original notebook: 10D for clustering, 2D for viz), then sweeps HDBSCAN
min_cluster_size to find a setting suited to this dataset's size (318 rows,
vs. the much larger original Scopus corpus that min_cluster_size=30 was tuned
for).

Input:  data/hsri/hsri_project_summaries.json (from step 1)
Output: data/hsri/hsri_with_embeddings.pkl (full, incl. embedding vectors)
        data/hsri/hsri_map_ready.csv (viz-ready, no embedding vectors)
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

IN_FILE = ROOT / "data" / "hsri" / "hsri_project_summaries.json"
OUT_PKL = ROOT / "data" / "hsri" / "hsri_with_embeddings.pkl"
OUT_CSV = ROOT / "data" / "hsri" / "hsri_map_ready.csv"

EMBEDDING_MODEL = "embed-multilingual-v3.0"
BATCH_SIZE = 48
SLEEP_TIME = 1.0
MAX_RETRIES = 5
MAX_CHARS = 4000  # generous safety cap only; real 512-token truncation is left to the
                   # Cohere API itself (truncate="END"), since Thai is far more token-dense
                   # per character than English and a char-based estimate here would be a
                   # guess -- English-first ordering means an END-truncation still eats the
                   # Thai tail first, which is the intent.

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")


def build_embedding_text(row):
    """English summary first, Thai source second -- so truncation at the
    model's token limit degrades the Thai tail rather than dropping English."""
    english_parts = [
        row.get("title_en", ""),
        row.get("summary_short", ""),
        row.get("main_findings", ""),
        row.get("practical_relevance", ""),
        ", ".join(row.get("key_concepts") or []),
    ]
    english_blob = " ".join(p for p in english_parts if isinstance(p, str) and p.strip())

    thai_parts = [
        row.get("project_title_th", ""),
        row.get("utilization_blob", ""),
    ]
    thai_blob = " ".join(p for p in thai_parts if isinstance(p, str) and p.strip())

    combined = (english_blob + "\n\n" + thai_blob).strip()
    return combined[:MAX_CHARS]


def report_length_distribution(texts):
    lengths = [len(t) for t in texts]
    s = pd.Series(lengths)
    print("Embedding text length (chars) distribution:")
    print(s.describe())
    print(f"  Rows over {MAX_CHARS} chars (truncated): {(s > MAX_CHARS).sum()} / {len(s)}")


def generate_embeddings_with_resume(texts, save_file, co):
    if save_file.exists():
        df_saved = pd.read_pickle(save_file)
        existing = df_saved["embedding"].tolist()
        print(f"Resuming from {len(existing)} existing embeddings")
    else:
        existing = []
        print("Starting fresh embedding generation")

    all_embeddings = existing.copy()
    start_idx = len(existing)

    for i in tqdm(range(start_idx, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i:i + BATCH_SIZE]
        retries = 0
        while retries < MAX_RETRIES:
            try:
                response = co.embed(
                    model=EMBEDDING_MODEL,
                    texts=batch,
                    input_type="search_document",
                    truncate="END",
                )
                all_embeddings.extend(response.embeddings)
                pd.DataFrame({"embedding": all_embeddings}).to_pickle(save_file)
                time.sleep(SLEEP_TIME)
                break
            except Exception as e:
                retries += 1
                wait = 10 * retries
                print(f"Batch {i} failed (attempt {retries}/{MAX_RETRIES}): {e}")
                if retries < MAX_RETRIES:
                    time.sleep(wait)
                else:
                    raise

    return all_embeddings


def sweep_hdbscan(X_cluster, sizes=(5, 8, 10, 15)):
    import hdbscan
    print("\nHDBSCAN min_cluster_size sweep (pre-refinement noise ratio is what matters):")
    results = []
    for size in sizes:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=size, min_samples=max(3, size // 3),
            cluster_selection_method="eom", metric="euclidean",
        )
        labels = clusterer.fit_predict(X_cluster)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = (labels == -1).mean()
        results.append((size, n_clusters, noise_ratio))
        print(f"  min_cluster_size={size:>3}: {n_clusters:>2} clusters, noise={noise_ratio:.1%}")
    return results


def main():
    if not COHERE_API_KEY:
        raise SystemExit(
            "COHERE_API_KEY not found in environment / .env -- cannot embed. "
            "Set it and re-run."
        )
    import cohere
    import umap

    co = cohere.Client(COHERE_API_KEY)

    df = pd.read_json(IN_FILE)
    print(f"Loaded {len(df)} projects")

    df["embedding_text"] = df.apply(build_embedding_text, axis=1)
    report_length_distribution(df["embedding_text"].tolist())

    texts = df["embedding_text"].tolist()
    embed_cache = ROOT / "data" / "hsri" / "hsri_embeddings_cache.pkl"
    embeddings = generate_embeddings_with_resume(texts, embed_cache, co)
    df["embedding"] = embeddings

    X = np.vstack(df["embedding"].values)
    print(f"Embedding matrix shape: {X.shape}")

    print("Reducing dimensions for clustering (10D)...")
    umap_cluster = umap.UMAP(
        n_neighbors=15, n_components=10, min_dist=0.0, metric="cosine", random_state=42,
    )
    X_cluster = umap_cluster.fit_transform(X)

    print("Creating 2D visualization coordinates...")
    umap_viz = umap.UMAP(
        n_neighbors=30, n_components=2, min_dist=0.1, metric="cosine", random_state=42,
    )
    X_viz = umap_viz.fit_transform(X)

    df["umap_x"] = X_viz[:, 0]
    df["umap_y"] = X_viz[:, 1]

    # keep the 10D cluster coords for step 3 (kNN refinement + density)
    df["coords_10d"] = list(X_cluster)

    sweep_hdbscan(X_cluster)

    df.to_pickle(OUT_PKL)
    df.drop(columns=["embedding", "coords_10d"]).to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_PKL} and {OUT_CSV}")


if __name__ == "__main__":
    main()
