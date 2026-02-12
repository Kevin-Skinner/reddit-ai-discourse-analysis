"""
Stage 08: UMAP Dimensionality Reduction
=========================================
We reduce the 768-dimensional Gemma embeddings to 3D using UMAP (Uniform
Manifold Approximation and Projection). This serves two purposes:
1. Visualization -- 3D coordinates can be rendered as an interactive point cloud
2. Clustering -- HDBSCAN works well in low-dimensional spaces

Our approach:
- Train UMAP on a balanced sample (50K posts + 50K comments = 100K total)
- Apply the fitted model to transform ALL documents
- Use cosine distance as the metric (preserves angular relationships from embedding space)
- Stream results in 50K chunks to avoid memory issues at scale
"""

import os
import numpy as np
import duckdb
import umap
from tqdm import tqdm
import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


CHUNK_SIZE = 50_000


def train_mapper(posts_db, comments_db, cfg):
    """Train UMAP on a mixed sample of posts and comments."""
    sample_size = cfg["umap"]["training_sample"]
    half = sample_size // 2

    con_p = duckdb.connect(posts_db, read_only=True)
    table_p = con_p.execute("SHOW TABLES").fetchone()[0]
    df_posts = con_p.execute(f"SELECT vector FROM {table_p} USING SAMPLE {half}").fetchdf()
    con_p.close()

    con_c = duckdb.connect(comments_db, read_only=True)
    table_c = con_c.execute("SHOW TABLES").fetchone()[0]
    df_comments = con_c.execute(f"SELECT vector FROM {table_c} USING SAMPLE {half}").fetchdf()
    con_c.close()

    X_train = np.vstack([
        np.stack(df_posts['vector'].values),
        np.stack(df_comments['vector'].values)
    ])
    print(f"Training UMAP on {len(X_train):,} samples...")

    mapper = umap.UMAP(
        n_components=cfg["umap"]["n_components"],
        n_neighbors=cfg["umap"]["n_neighbors"],
        min_dist=cfg["umap"]["min_dist"],
        metric=cfg["umap"]["metric"],
        low_memory=True,
        verbose=True
    ).fit(X_train)

    return mapper


def transform_and_save(mapper, source_db, type_label, output_con):
    """Stream data from a vectors DB, transform to 3D, and save."""
    con_read = duckdb.connect(source_db, read_only=True)
    table_name = con_read.execute("SHOW TABLES").fetchone()[0]
    total = con_read.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    pbar = tqdm(total=total, desc=f"Mapping {type_label}")
    offset = 0

    while offset < total:
        df = con_read.execute(f"SELECT id, vector FROM {table_name} LIMIT {CHUNK_SIZE} OFFSET {offset}").fetchdf()
        if df.empty:
            break

        vectors = np.stack(df['vector'].values)
        coords_3d = mapper.transform(vectors)

        rows = [
            (df['id'].iloc[i], float(coords_3d[i][0]), float(coords_3d[i][1]),
             float(coords_3d[i][2]), type_label)
            for i in range(len(df))
        ]
        output_con.executemany("INSERT INTO embeddings_3d VALUES (?, ?, ?, ?, ?)", rows)

        offset += CHUNK_SIZE
        pbar.update(len(df))

    con_read.close()
    pbar.close()


def main():
    cfg = load_config()
    output_db = cfg["duckdb"]["embeddings_3d_db"]

    # Create output database
    if os.path.exists(output_db):
        os.remove(output_db)
    con_out = duckdb.connect(output_db)
    con_out.execute("""
        CREATE TABLE embeddings_3d (
            id VARCHAR, x FLOAT, y FLOAT, z FLOAT, type VARCHAR
        )
    """)

    # Train UMAP on a balanced sample
    mapper = train_mapper(
        cfg["duckdb"]["vectors_posts_db"],
        cfg["duckdb"]["vectors_comments_db"],
        cfg
    )

    # Transform all documents
    transform_and_save(mapper, cfg["duckdb"]["vectors_posts_db"], "post", con_out)
    transform_and_save(mapper, cfg["duckdb"]["vectors_comments_db"], "comment", con_out)

    print(f"3D map saved to: {output_db}")
    con_out.close()


if __name__ == "__main__":
    main()

