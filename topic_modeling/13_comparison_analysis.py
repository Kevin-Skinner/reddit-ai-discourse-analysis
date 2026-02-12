"""
Stage 13: Comparison Analysis
===============================
We compare subreddits and clusters across several dimensions:

1. **Topic overlap**: How many topics do two subreddits share? We measure this
   via Jaccard similarity of topic keyword sets.
2. **Cluster distribution**: How are a subreddit's documents distributed across
   HDBSCAN clusters? A subreddit concentrated in one cluster is thematically
   focused; one spread across many clusters is diverse.
3. **Cluster diversity**: Shannon entropy of each cluster's subreddit distribution.
   High diversity means the cluster draws from many communities (a shared theme);
   low diversity means it's dominated by one subreddit.
4. **Visualization**: Heatmaps, bar charts, and network graphs to explore these
   relationships.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

from bertopic_modeling import load_topic_model, get_all_topics


# ---- Topic similarity ----

def topic_similarity_jaccard(words1, words2):
    """Jaccard similarity between two topic keyword lists."""
    s1 = {w[0].lower() for w in words1}
    s2 = {w[0].lower() for w in words2}
    union = s1 | s2
    return len(s1 & s2) / len(union) if union else 0.0


def find_shared_topics(model1, model2, threshold=0.3):
    """Find topic pairs shared between two BERTopic models."""
    t1 = get_all_topics(model1)
    t2 = get_all_topics(model2)
    shared = []
    for id1, w1 in t1.items():
        if id1 == -1:
            continue
        for id2, w2 in t2.items():
            if id2 == -1:
                continue
            sim = topic_similarity_jaccard(w1, w2)
            if sim >= threshold:
                shared.append((id1, id2, sim))
    return sorted(shared, key=lambda x: x[2], reverse=True)


def topic_overlap_matrix(subreddits, models_dir, threshold=0.3):
    """Build a pairwise topic-overlap matrix for a list of subreddits."""
    n = len(subreddits)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            safe_i = subreddits[i].replace("/", "_").replace("\\", "_")
            safe_j = subreddits[j].replace("/", "_").replace("\\", "_")
            pi = os.path.join(models_dir, f"subreddit_{safe_i}.pkl")
            pj = os.path.join(models_dir, f"subreddit_{safe_j}.pkl")
            if os.path.exists(pi) and os.path.exists(pj):
                m1 = load_topic_model(pi)
                m2 = load_topic_model(pj)
                shared = find_shared_topics(m1, m2, threshold)
                n1 = len([t for t in get_all_topics(m1) if t != -1])
                n2 = len([t for t in get_all_topics(m2) if t != -1])
                score = len(shared) / max(n1, n2) if max(n1, n2) > 0 else 0.0
                mat[i, j] = mat[j, i] = score
    return pd.DataFrame(mat, index=subreddits, columns=subreddits)


# ---- Cluster / subreddit distributions ----

def subreddit_cluster_distribution(df, subreddit):
    """Distribution of a subreddit's documents across clusters."""
    return df[df["subreddit"] == subreddit]["cluster_id"].value_counts().sort_index()


def cluster_subreddit_distribution(df, cluster_id):
    """Distribution of subreddits within a cluster."""
    return df[df["cluster_id"] == cluster_id]["subreddit"].value_counts()


def cluster_diversity_score(df, cluster_id):
    """
    Normalized Shannon entropy of the subreddit distribution within a cluster.
    Returns a value in [0, 1]: 0 = single subreddit, 1 = perfectly uniform.
    """
    sub = df[df["cluster_id"] == cluster_id]
    if len(sub) == 0:
        return 0.0
    counts = sub["subreddit"].value_counts()
    probs = counts / len(sub)
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    max_ent = np.log2(len(counts)) if len(counts) > 1 else 1.0
    return entropy / max_ent if max_ent > 0 else 0.0


def get_cluster_subreddit_matrix(df, top_clusters=None, top_subreddits=None):
    """Build a (clusters x subreddits) count matrix."""
    if top_clusters is None:
        top_clusters = sorted(c for c in df["cluster_id"].unique() if c != -1)
    if top_subreddits is None:
        top_subreddits = df["subreddit"].value_counts().head(20).index.tolist()
    rows = []
    for cid in top_clusters:
        cdf = df[df["cluster_id"] == cid]
        row = {"cluster_id": cid}
        for sub in top_subreddits:
            row[sub] = len(cdf[cdf["subreddit"] == sub])
        rows.append(row)
    return pd.DataFrame(rows).set_index("cluster_id")


# ---- Visualizations ----

def plot_overlap_heatmap(matrix, output_path=None, title="Topic Overlap Between Subreddits"):
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cluster_distribution(matrix, output_path=None, top_n=15,
                               title="Subreddit Distribution Across Clusters"):
    top_subs = matrix.sum().nlargest(top_n).index
    plt.figure(figsize=(14, 8))
    matrix[top_subs].plot(kind="bar", stacked=True, colormap="tab20", width=0.8)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Cluster ID")
    plt.ylabel("Document Count")
    plt.legend(title="Subreddit", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_diversity_scores(df, top_n=20, output_path=None,
                           title="Cluster Diversity Scores (Shannon Entropy)"):
    cids = sorted(c for c in df["cluster_id"].unique() if c != -1)[:top_n]
    scores = [cluster_diversity_score(df, c) for c in cids]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(cids)), scores, color="steelblue", alpha=0.7)
    plt.xticks(range(len(cids)), cids, rotation=45, ha="right")
    plt.xlabel("Cluster ID")
    plt.ylabel("Diversity Score")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

