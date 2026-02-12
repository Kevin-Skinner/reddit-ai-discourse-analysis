"""
Stage 11: Data Integration
============================
We merge all upstream outputs into a single unified DataFrame for topic modeling:
  - 3D UMAP coordinates (from Stage 08)
  - HDBSCAN cluster assignments (from Stage 09)
  - Original text content (from the tokenized databases)
  - Subreddit labels (from the source databases)
  - Original 768D Gemma embeddings (from Stage 07, for BERTopic input)

This integration layer allows downstream stages to work with a single DataFrame
containing everything needed for topic modeling and comparison analysis.
"""

import os
import duckdb
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_table_name(con, schema_name="reddit"):
    """Find the first table in a schema."""
    meta = con.execute(
        f"SELECT table_schema, table_name FROM information_schema.tables "
        f"WHERE table_type='BASE TABLE' AND table_schema='{schema_name}' LIMIT 1"
    ).fetchone()
    return f"{meta[0]}.{meta[1]}" if meta else None


def extract_text(db_path, ids, content_type="post"):
    """
    Extract text content and subreddit names for a list of document IDs.
    For posts: concatenates title + selftext.
    For comments: uses the body field.
    """
    if not ids:
        return pd.DataFrame(columns=["id", "subreddit", "text"])

    con = duckdb.connect(db_path, read_only=True)
    table_name = get_table_name(con)
    if not table_name:
        con.close()
        return pd.DataFrame(columns=["id", "subreddit", "text"])

    text_expr = (
        "COALESCE(title, '') || ' ' || COALESCE(selftext, '')"
        if content_type == "post" else "COALESCE(body, '')"
    )

    # Process in batches to avoid SQL string limits
    results = []
    batch_size = 10_000
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        id_str = "'" + "','".join(batch) + "'"
        df = con.execute(
            f"SELECT id, subreddit, {text_expr} AS text FROM {table_name} WHERE id IN ({id_str})"
        ).fetchdf()
        if not df.empty:
            results.append(df)

    con.close()
    if results:
        df = pd.concat(results, ignore_index=True)
        df["text"] = df["text"].astype(str).str.strip().replace("", np.nan)
        return df.dropna(subset=["text"])
    return pd.DataFrame(columns=["id", "subreddit", "text"])


def load_combined_dataset(cfg, sample_size=None):
    """
    Merge embeddings, clusters, text, and subreddit data into one DataFrame.

    Returns a DataFrame with columns:
        id, type, x, y, z, cluster_id, subreddit, text
    """
    # 1. Load 3D coordinates
    con = duckdb.connect(cfg["duckdb"]["embeddings_3d_db"], read_only=True)
    df_coords = con.execute("SELECT id, type, x, y, z FROM embeddings_3d").fetchdf()
    con.close()
    print(f"Loaded {len(df_coords):,} embeddings")

    # 2. Load cluster assignments
    assignments_path = os.path.join(cfg["output"]["results_dir"], "cluster_assignments.parquet")
    df_clusters = pd.read_parquet(assignments_path)

    # 3. Merge
    df = pd.merge(df_coords, df_clusters, on="id", how="inner")

    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    # 4. Extract text + subreddit
    ids_p = df[df["type"] == "post"]["id"].tolist()
    ids_c = df[df["type"] == "comment"]["id"].tolist()

    df_posts = extract_text(cfg["duckdb"]["tokenized_posts_db"], ids_p, "post")
    df_comments = extract_text(cfg["duckdb"]["tokenized_comments_db"], ids_c, "comment")
    df_text = pd.concat([df_posts, df_comments], ignore_index=True)

    df = pd.merge(df, df_text, on="id", how="inner")
    df["subreddit"] = df["subreddit"].fillna("Unknown")
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    print(f"Combined dataset: {len(df):,} documents")
    return df


# --- Filtering helpers ---

def filter_by_cluster(df, cluster_id):
    return df[df["cluster_id"] == cluster_id].copy()

def filter_by_subreddit(df, subreddit):
    return df[df["subreddit"] == subreddit].copy()

def get_top_clusters(df, top_n=20, exclude_noise=True):
    counts = df["cluster_id"].value_counts()
    if exclude_noise and -1 in counts.index:
        counts = counts.drop(-1)
    return counts.head(top_n).index.tolist()

def get_top_subreddits(df, top_n=50):
    counts = df["subreddit"].value_counts()
    for label in ["Unknown", None]:
        if label in counts.index:
            counts = counts.drop(label)
    return counts.head(top_n).index.tolist()


# --- Gemma embedding retrieval ---

def get_gemma_embeddings_batch(ids, types, cfg, batch_size=10_000):
    """
    Load original 768D Gemma embeddings for a list of documents.
    Groups by content type and queries the appropriate vectors database.

    Returns:
        (embeddings_array, valid_indices) -- aligned with input order
    """
    post_ids, post_idx = [], []
    comment_ids, comment_idx = [], []
    for i, (doc_id, doc_type) in enumerate(zip(ids, types)):
        if doc_type == "post":
            post_ids.append(doc_id)
            post_idx.append(i)
        else:
            comment_ids.append(doc_id)
            comment_idx.append(i)

    id_to_vec = {}

    def _load(db_path, table_name, id_list):
        if not id_list or not os.path.exists(db_path):
            return
        con = duckdb.connect(db_path, read_only=True)
        for i in range(0, len(id_list), batch_size):
            batch = id_list[i:i + batch_size]
            id_str = "'" + "','".join(batch) + "'"
            rows = con.execute(f"SELECT id, vector FROM {table_name} WHERE id IN ({id_str})").fetchall()
            for row in rows:
                id_to_vec[row[0]] = np.array(row[1])
        con.close()

    _load(cfg["duckdb"]["vectors_posts_db"], "posts_vectors", post_ids)
    _load(cfg["duckdb"]["vectors_comments_db"], "comments_vectors", comment_ids)

    valid_indices, embeddings = [], []
    for i, doc_id in enumerate(ids):
        if doc_id in id_to_vec:
            valid_indices.append(i)
            embeddings.append(id_to_vec[doc_id])

    if not embeddings:
        return np.array([]), []
    return np.array(embeddings), valid_indices


if __name__ == "__main__":
    cfg = load_config()
    df = load_combined_dataset(cfg, sample_size=1000)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Top clusters:\n{df['cluster_id'].value_counts().head(5)}")
    print(f"Top subreddits:\n{df['subreddit'].value_counts().head(5)}")

