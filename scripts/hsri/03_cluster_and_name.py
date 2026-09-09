"""
Step 3 (HSRI adaptation of 03_Clustering_and_Name_Assignment.ipynb)
======================================================================

Reuses the parts of the original notebook that are actually reproducible from
this repo's own pipeline (HDBSCAN + kNN noise refinement, local-density
classification), applied to step 2's own 10D UMAP coordinates -- NOT the
external 5-layer hierarchy file the original notebook loaded from Google
Drive, which was never part of this repo and isn't reconstructable here.
Produces a SINGLE clustering layer, named via OpenRouter (google/gemini-2.5-flash)
instead of OpenAI.

Input:  data/hsri/hsri_with_embeddings.pkl (from step 2)
Output: data/hsri/hsri_clustered_named.pkl / .csv
"""
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from scipy.spatial.distance import cdist
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

IN_FILE = ROOT / "data" / "hsri" / "hsri_with_embeddings.pkl"
OUT_PKL = ROOT / "data" / "hsri" / "hsri_clustered_named.pkl"
OUT_CSV = ROOT / "data" / "hsri" / "hsri_clustered_named.csv"

MIN_CLUSTER_SIZE = int(os.environ.get("HSRI_MIN_CLUSTER_SIZE", "8"))  # set from step 2's sweep
MODEL = "google/gemini-2.5-flash"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


def perform_hdbscan_knn_clustering(X, min_cluster_size):
    import hdbscan

    print("Running initial HDBSCAN clustering...")
    # min_samples matches the sweep in step 2 (default min_samples=min_cluster_size is far
    # too conservative here and collapses the cluster count).
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(3, min_cluster_size // 3),
        cluster_selection_method="eom",
        metric="euclidean",
        gen_min_span_tree=True,
    )
    initial_labels = clusterer.fit_predict(X)

    core_mask = initial_labels != -1
    noise_mask = initial_labels == -1
    n_clusters = len(set(initial_labels)) - (1 if -1 in initial_labels else 0)
    noise_ratio = noise_mask.mean()
    print(f"  clusters={n_clusters}, pre-refinement noise ratio={noise_ratio:.1%}")

    final_labels = initial_labels.copy()

    if noise_mask.any() and core_mask.any():
        print("Refining noise points with kNN...")
        knn = KNeighborsClassifier(n_neighbors=min(15, core_mask.sum()), weights="distance")
        knn.fit(X[core_mask], initial_labels[core_mask])
        final_labels[noise_mask] = knn.predict(X[noise_mask])

    refined_noise_ratio = (final_labels == -1).mean()
    print(f"  post-refinement noise ratio={refined_noise_ratio:.1%} (should be ~0%)")
    return final_labels, initial_labels


def analyze_density(X, k=16):
    nn = NearestNeighbors(n_neighbors=min(k, len(X))).fit(X)
    distances, _ = nn.kneighbors(X)
    density_scores = np.sum(1.0 / (distances[:, 1:] + 1e-5), axis=1)
    return density_scores


def get_representative_projects(cluster_df, n=15):
    if len(cluster_df) == 0:
        return []
    coords = np.vstack(cluster_df["coords_10d"].values)
    centroid = coords.mean(axis=0).reshape(1, -1)
    distances = cdist(coords, centroid, metric="euclidean").flatten()
    dmin, dmax = distances.min(), distances.max()
    norm_dist = (distances - dmin) / (dmax - dmin) if dmax > dmin else np.zeros_like(distances)
    scores = 1 - norm_dist
    top_idx = scores.argsort()[-n:][::-1]
    return cluster_df.iloc[top_idx]["project_id"].tolist()


NAMING_PROMPT = """
Analyze these {n} Thai health-systems research project summaries from a single cluster:

{context}

Generate:
1. A concise, descriptive cluster name (max 8 words, English, title case)
2. A one-sentence description of this cluster's research scope

Return only a JSON object: {{"name": "...", "description": "..."}}
"""


def generate_cluster_name(client, summaries):
    context = "\n---\n".join(summaries[:15])
    prompt = NAMING_PROMPT.format(n=len(summaries[:15]), context=context)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        print(f"  naming error: {e}")
        return {"name": "Unnamed Cluster", "description": "Description not available."}


def main():
    if not OPENROUTER_API_KEY:
        raise SystemExit("OPENROUTER_API_KEY not found in environment / .env")

    df = pd.read_pickle(IN_FILE)
    print(f"Loaded {len(df)} projects")

    X = np.vstack(df["coords_10d"].values)
    final_labels, initial_labels = perform_hdbscan_knn_clustering(X, MIN_CLUSTER_SIZE)
    df["cluster_label"] = final_labels
    df["original_label"] = initial_labels

    density_scores = analyze_density(X)
    df["density_score"] = density_scores

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

    naming_results = []
    for cluster_id, cluster_group in tqdm(df.groupby("cluster_label"), desc="Naming clusters"):
        if cluster_id == -1:
            continue
        rep_ids = get_representative_projects(cluster_group)
        summaries = df[df["project_id"].isin(rep_ids)]["summary_short"].dropna().tolist()
        if not summaries:
            continue
        meta = generate_cluster_name(client, summaries)
        naming_results.append({"cluster_label": cluster_id, **meta})
        time.sleep(0.5)

    naming_df = pd.DataFrame(naming_results)
    name_map = naming_df.set_index("cluster_label")["name"].to_dict()
    desc_map = naming_df.set_index("cluster_label")["description"].to_dict()

    df["cluster_name"] = df["cluster_label"].map(name_map).fillna("Unclustered")
    df["cluster_description"] = df["cluster_label"].map(desc_map).fillna("")

    df.to_pickle(OUT_PKL)
    df.drop(columns=["embedding", "coords_10d"]).to_csv(OUT_CSV, index=False)

    print(f"\nNamed {len(naming_results)} clusters")
    print(df["cluster_name"].value_counts())
    print(f"\nSaved {OUT_PKL} and {OUT_CSV}")


if __name__ == "__main__":
    main()
