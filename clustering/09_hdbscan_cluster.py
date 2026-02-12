"""
Stage 09: HDBSCAN Clustering
==============================
We cluster the 3D UMAP embeddings using HDBSCAN (Hierarchical Density-Based
Spatial Clustering of Applications with Noise). HDBSCAN is ideal here because:

1. It doesn't require specifying k (number of clusters) upfront
2. It naturally identifies noise points -- documents that don't belong to any
   dense region -- which is important given the diversity of Reddit content
3. It finds clusters of varying density, capturing both large mainstream
   topics and small niche communities

Outputs:
- cluster_assignments.parquet: Maps each document ID to its cluster label (-1 = noise)
- cluster_centroids.csv: Mean (x, y, z) coordinates for each cluster
"""

import os
import duckdb
import hdbscan
import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_clusters(cfg):
    input_db = cfg["duckdb"]["embeddings_3d_db"]
    output_dir = cfg["output"]["results_dir"]
    os.makedirs(output_dir, exist_ok=True)

    assignments_path = os.path.join(output_dir, "cluster_assignments.parquet")
    centroids_path = os.path.join(output_dir, "cluster_centroids.csv")

    # Load all 3D coordinates
    con = duckdb.connect(input_db, read_only=True)
    df = con.execute("SELECT id, x, y, z FROM embeddings_3d").fetchdf()
    con.close()
    print(f"Loaded {len(df):,} documents")

    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg["hdbscan"]["min_cluster_size"],
        min_samples=cfg["hdbscan"]["min_samples"],
        metric=cfg["hdbscan"].get("metric", "euclidean"),
        core_dist_n_jobs=-1  # Use all CPU cores
    )
    clusters = clusterer.fit_predict(df[["x", "y", "z"]])

    # Save cluster assignments
    df_assignments = df[["id"]].copy()
    df_assignments["cluster_id"] = clusters
    df_assignments.to_parquet(assignments_path, index=False)

    # Compute and save cluster centroids (excluding noise)
    df_with_clusters = df.copy()
    df_with_clusters["cluster_id"] = clusters
    df_clean = df_with_clusters[df_with_clusters["cluster_id"] != -1]

    df_centroids = df_clean.groupby("cluster_id")[["x", "y", "z"]].mean().reset_index()
    df_centroids.to_csv(centroids_path, index=False)

    n_clusters = df_assignments["cluster_id"].max() + 1
    n_noise = (df_assignments["cluster_id"] == -1).sum()
    print(f"Results: {n_clusters:,} clusters, {n_noise:,} noise points ({100*n_noise/len(df):.1f}%)")
    print(f"Saved to: {assignments_path}")


def main():
    cfg = load_config()
    generate_clusters(cfg)


if __name__ == "__main__":
    main()

