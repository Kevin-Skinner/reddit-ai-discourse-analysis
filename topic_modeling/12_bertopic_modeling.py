"""
Stage 12: BERTopic Topic Modeling
==================================
We train BERTopic models at two levels of granularity:

1. **Per cluster**: For each of the top-N HDBSCAN clusters, we train a separate
   BERTopic model using the original 768D Gemma embeddings. This reveals the
   sub-topics within each semantically coherent cluster.

2. **Per subreddit**: For each of the top-M subreddits by document count, we train
   a BERTopic model to capture community-specific discourse patterns.

Key design decisions:
- We pass pre-computed Gemma embeddings to BERTopic (embedding_model=None),
  so it uses our domain-specific embeddings rather than a generic sentence-transformer
- Class-based TF-IDF with BM25 weighting for robust topic representations
- Maximal Marginal Relevance (MMR) for diverse keyword selection
- Topic centroids (mean Gemma embedding per topic, L2-normalized) are computed
  and saved for downstream similarity analysis
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import yaml
import warnings
warnings.filterwarnings("ignore")


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_bertopic_model(cfg):
    """
    Create a configured BERTopic model using parameters from config.

    We configure each sub-component explicitly:
    - UMAP for internal dimensionality reduction (separate from our pipeline UMAP)
    - HDBSCAN for internal clustering
    - BM25-weighted class TF-IDF for topic representation
    - MMR for keyword diversity
    """
    bt_cfg = cfg.get("bertopic", {})

    umap_model = UMAP(
        n_neighbors=bt_cfg.get("n_neighbors", 30),
        n_components=bt_cfg.get("n_components", 5),
        min_dist=0.0, metric="cosine", random_state=42
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=cfg["hdbscan"]["min_cluster_size"],
        min_samples=cfg["hdbscan"]["min_samples"],
        prediction_data=True, metric="euclidean", core_dist_n_jobs=-1
    )
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
    representation_model = MaximalMarginalRelevance(
        diversity=bt_cfg.get("diversity", 0.5)
    )
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3), min_df=2)

    return BERTopic(
        calculate_probabilities=False,
        representation_model=representation_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=None,          # We supply our own Gemma embeddings
        top_n_words=bt_cfg.get("top_n_words", 10),
        min_topic_size=bt_cfg.get("min_topic_size", 20),
        ctfidf_model=ctfidf_model,
        vectorizer_model=vectorizer_model,
        verbose=True
    )


# --- Centroid computation ---

def _compute_centroids(topics, embeddings, scope_type, scope_value, extra_cols=None):
    """
    Compute L2-normalized mean embedding per topic (excluding noise topic -1).
    Returns a DataFrame of centroid vectors.
    """
    topics = np.asarray(topics).flatten()
    valid = topics != -1
    if not valid.any():
        return pd.DataFrame()

    E_valid = embeddings[valid]
    t_valid = topics[valid]
    rows = []

    for t in sorted(set(int(x) for x in t_valid)):
        mask = t_valid == t
        E_t = E_valid[mask]
        centroid = E_t.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        row = {
            "scope_type": scope_type,
            "scope_value": scope_value,
            "topic_id": int(t),
            "n_docs": int(mask.sum()),
            "centroid": centroid.tolist(),
        }
        if extra_cols:
            row.update(extra_cols)
        rows.append(row)

    return pd.DataFrame(rows)


def _append_centroids(df_new, path, scope_type, scope_value):
    """Append centroid rows to a Parquet file, replacing any existing rows for the same scope."""
    if df_new.empty:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        df_old = pd.read_parquet(path)
        df_old = df_old[~(
            (df_old["scope_type"] == scope_type) & (df_old["scope_value"] == scope_value)
        )]
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_parquet(path, index=False)


# --- Training functions ---

def train_cluster_model(cluster_id, docs, embeddings, cfg,
                        doc_ids=None, subreddits=None):
    """
    Train a BERTopic model for a single HDBSCAN cluster.

    Args:
        cluster_id: Cluster label
        docs: Document texts
        embeddings: 768D Gemma embeddings (n_docs x 768)
        cfg: Config dict
        doc_ids: Document IDs (for centroid tracking)
        subreddits: Subreddit labels (for centroid metadata)

    Returns:
        (model, topics, topic_info) or (None, [], {}) if too few docs
    """
    min_docs = cfg["analysis"].get("min_docs_per_model", 100)
    if len(docs) < min_docs or embeddings.shape[0] != len(docs):
        return None, [], {}

    model = create_bertopic_model(cfg)
    topics, _ = model.fit_transform(docs, embeddings=embeddings)
    topic_info = model.get_topic_info()

    # Save model
    models_dir = cfg["output"]["topic_models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"cluster_{cluster_id}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Compute and save centroids
    if doc_ids is not None and subreddits is not None:
        centroids_path = os.path.join(cfg["output"]["data_exports_dir"], "topic_centroids_clusters.parquet")
        top_subs = (
            pd.Series(subreddits).value_counts().head(5).index.tolist()
        )
        df_c = _compute_centroids(
            topics, embeddings, "cluster", str(cluster_id),
            extra_cols={"cluster_id": cluster_id, "top_subreddits": top_subs}
        )
        _append_centroids(df_c, centroids_path, "cluster", str(cluster_id))

    return model, topics, topic_info


def train_subreddit_model(subreddit, docs, embeddings, cfg, doc_ids=None):
    """
    Train a BERTopic model for a single subreddit.

    Args:
        subreddit: Subreddit name
        docs: Document texts
        embeddings: 768D Gemma embeddings
        cfg: Config dict
        doc_ids: Document IDs (for centroid tracking)

    Returns:
        (model, topics, topic_info) or (None, [], {}) if too few docs
    """
    min_docs = cfg["analysis"].get("min_docs_per_model", 100)
    if len(docs) < min_docs:
        return None, [], {}
    if embeddings is None or embeddings.shape[0] != len(docs):
        return None, [], {}

    model = create_bertopic_model(cfg)
    topics, _ = model.fit_transform(docs, embeddings=embeddings)
    topic_info = model.get_topic_info()

    # Save model
    models_dir = cfg["output"]["topic_models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    safe_name = subreddit.replace("/", "_").replace("\\", "_")
    model_path = os.path.join(models_dir, f"subreddit_{safe_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Compute and save centroids
    if doc_ids is not None:
        centroids_path = os.path.join(cfg["output"]["data_exports_dir"], "topic_centroids_subreddits.parquet")
        df_c = _compute_centroids(topics, embeddings, "subreddit", subreddit)
        _append_centroids(df_c, centroids_path, "subreddit", subreddit)

    return model, topics, topic_info


def load_topic_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def get_all_topics(model):
    return model.get_topics()


def extract_topic_words(model, topic_id, top_n=10):
    words = model.get_topic(topic_id)
    return words[:top_n] if words else []

