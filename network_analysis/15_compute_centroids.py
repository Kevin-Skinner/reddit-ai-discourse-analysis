"""
Stage 15: Compute 3D Coordinates for Topic Centroids
======================================================
We take the 768-dimensional topic centroid vectors (computed in Stage 12 for
each cluster and subreddit topic model) and project them into a shared 3D
space using UMAP. We also compute pairwise cosine similarity between all
centroids to build an edge list for network visualization.

This lets us visualize where topics live relative to each other in semantic space:
- Similar topics cluster together
- Cross-community topics (topics that appear in multiple subreddits) form bridges
- The edge list captures which topics are semantically related

Outputs:
- topic_centroids_3d.parquet: Centroids with x, y, z coordinates
- topic_similarity_edges.parquet: Pairwise edges above the similarity threshold
"""

import os
import pandas as pd
import numpy as np
from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import warnings
warnings.filterwarnings("ignore")


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_centroids(cfg):
    """Load cluster and subreddit centroids from their respective Parquet files."""
    export_dir = cfg["output"]["data_exports_dir"]
    frames = []

    # Cluster centroids
    cluster_path = os.path.join(export_dir, "topic_centroids_clusters.parquet")
    if os.path.exists(cluster_path):
        df = pd.read_parquet(cluster_path)
        if "scope_type" not in df.columns:
            df["scope_type"] = "cluster"
            df["scope_value"] = df["cluster_id"].astype(str)
        frames.append(df)
        print(f"  Loaded {len(df)} cluster centroids")

    # Subreddit centroids
    sub_path = os.path.join(export_dir, "topic_centroids_subreddits.parquet")
    if os.path.exists(sub_path):
        df = pd.read_parquet(sub_path)
        frames.append(df)
        print(f"  Loaded {len(df)} subreddit centroids")

    if not frames:
        raise FileNotFoundError("No centroid files found.")

    df_all = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(df_all)} centroids")
    return df_all


def compute_3d(vectors, cfg):
    """UMAP reduction of centroid vectors to 3 dimensions."""
    umap_cfg = cfg["umap"]
    reducer = UMAP(
        n_components=3,
        n_neighbors=umap_cfg["n_neighbors"],
        min_dist=umap_cfg["min_dist"],
        metric=umap_cfg["metric"],
        random_state=42, verbose=True
    )
    return reducer.fit_transform(vectors)


def compute_edges(df, vectors, threshold):
    """Build an edge list from pairwise cosine similarity above a threshold."""
    sim = cosine_similarity(vectors)
    n = len(vectors)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                ri, rj = df.iloc[i], df.iloc[j]
                edges.append({
                    "source_scope": ri.get("scope_type", "unknown"),
                    "source_value": str(ri.get("scope_value", "")),
                    "source_topic": int(ri["topic_id"]),
                    "target_scope": rj.get("scope_type", "unknown"),
                    "target_value": str(rj.get("scope_value", "")),
                    "target_topic": int(rj["topic_id"]),
                    "similarity": float(sim[i, j]),
                })
    print(f"  {len(edges)} edges above threshold {threshold}")
    return pd.DataFrame(edges)


def main():
    cfg = load_config()
    export_dir = cfg["output"]["data_exports_dir"]
    os.makedirs(export_dir, exist_ok=True)
    threshold = cfg["analysis"].get("similarity_threshold", 0.5)

    df_centroids = load_centroids(cfg)

    # Extract vectors
    vectors = np.array([
        c if isinstance(c, list) else np.array(c).tolist()
        for c in df_centroids["centroid"]
    ])
    print(f"  Vector matrix shape: {vectors.shape}")

    # UMAP to 3D
    coords = compute_3d(vectors, cfg)
    df_centroids["x"] = coords[:, 0]
    df_centroids["y"] = coords[:, 1]
    df_centroids["z"] = coords[:, 2]

    out_3d = os.path.join(export_dir, "topic_centroids_3d.parquet")
    df_centroids.to_parquet(out_3d, index=False)
    print(f"  Saved 3D centroids: {out_3d}")

    # Similarity edges
    df_edges = compute_edges(df_centroids, vectors, threshold)
    if not df_edges.empty:
        out_edges = os.path.join(export_dir, "topic_similarity_edges.parquet")
        df_edges.to_parquet(out_edges, index=False)
        print(f"  Saved edges: {out_edges}")


if __name__ == "__main__":
    main()

