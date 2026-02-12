"""
Stage 14: Run Topic Modeling Analysis (Orchestrator)
=====================================================
This is the main driver script for the topic modeling phase. It:

1. Loads the integrated dataset (embeddings + clusters + text + subreddits)
2. Identifies the top-N clusters and top-M subreddits by document count
3. Trains per-cluster BERTopic models using Gemma embeddings
4. Trains per-subreddit BERTopic models
5. Performs topic overlap analysis between subreddit pairs
6. Computes cluster diversity scores (Shannon entropy)
7. Generates visualization outputs and summary statistics

All outputs are saved to the configured output directory.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings("ignore")

# Ensure sibling modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_integration import (  # noqa: E402  (11_data_integration)
    load_combined_dataset, filter_by_cluster, filter_by_subreddit,
    get_top_clusters, get_top_subreddits, get_gemma_embeddings_batch
)
from bertopic_modeling import (  # noqa: E402  (12_bertopic_modeling)
    train_cluster_model, train_subreddit_model
)
from comparison_analysis import (  # noqa: E402  (13_comparison_analysis)
    topic_overlap_matrix, get_cluster_subreddit_matrix,
    cluster_diversity_score, plot_overlap_heatmap,
    plot_cluster_distribution, plot_diversity_scores
)


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---- Cluster model training ----

def train_all_cluster_models(df, cluster_ids, cfg):
    """Train BERTopic models for each of the top clusters."""
    results = {}
    min_docs = cfg["analysis"]["min_docs_per_model"]
    t0 = time.time()

    for i, cid in enumerate(cluster_ids, 1):
        print(f"\n  [{i}/{len(cluster_ids)}] Cluster {cid}")
        cdf = filter_by_cluster(df, cid)
        if len(cdf) < min_docs:
            print(f"    Skipped ({len(cdf)} docs < {min_docs})")
            continue

        docs = cdf["text"].tolist()
        ids = cdf["id"].tolist()
        types = cdf["type"].tolist()
        subs = cdf["subreddit"].tolist()

        embeddings, valid_idx = get_gemma_embeddings_batch(ids, types, cfg)
        if len(valid_idx) < min_docs:
            print(f"    Skipped (only {len(valid_idx)} embeddings)")
            continue

        docs = [docs[j] for j in valid_idx]
        ids = [ids[j] for j in valid_idx]
        subs = [subs[j] for j in valid_idx]

        model, topics, info = train_cluster_model(
            cid, docs, embeddings, cfg, doc_ids=ids, subreddits=subs
        )
        if model:
            n_topics = len([t for t in set(topics) if t != -1])
            results[cid] = {"model": model, "topics": topics,
                            "topic_info": info, "n_docs": len(docs),
                            "n_topics": n_topics}
            print(f"    {n_topics} topics found")

    print(f"\n  Trained {len(results)}/{len(cluster_ids)} cluster models "
          f"in {(time.time()-t0)/60:.1f} min")
    return results


# ---- Subreddit model training ----

def train_all_subreddit_models(df, subreddits, cfg):
    """Train BERTopic models for each of the top subreddits."""
    results = {}
    min_docs = cfg["analysis"]["min_docs_per_model"]
    t0 = time.time()

    for i, sub in enumerate(subreddits, 1):
        print(f"\n  [{i}/{len(subreddits)}] Subreddit: {sub}")
        sdf = filter_by_subreddit(df, sub)
        if len(sdf) < min_docs:
            print(f"    Skipped ({len(sdf)} docs < {min_docs})")
            continue

        docs = sdf["text"].tolist()
        ids = sdf["id"].tolist()
        types = sdf["type"].tolist()

        embeddings, valid_idx = get_gemma_embeddings_batch(ids, types, cfg)
        if len(valid_idx) < min_docs:
            print(f"    Skipped (only {len(valid_idx)} embeddings)")
            continue

        docs = [docs[j] for j in valid_idx]
        ids = [ids[j] for j in valid_idx]

        model, topics, info = train_subreddit_model(
            sub, docs, embeddings, cfg, doc_ids=ids
        )
        if model:
            n_topics = len([t for t in set(topics) if t != -1])
            results[sub] = {"model": model, "topics": topics,
                            "topic_info": info, "n_docs": len(docs),
                            "n_topics": n_topics}
            print(f"    {n_topics} topics found")

    print(f"\n  Trained {len(results)}/{len(subreddits)} subreddit models "
          f"in {(time.time()-t0)/60:.1f} min")
    return results


# ---- Main ----

def main():
    cfg = load_config()
    analysis = cfg["analysis"]

    # Create output directories
    for d in [cfg["output"]["results_dir"], cfg["output"]["topic_models_dir"],
              cfg["output"]["visualizations_dir"], cfg["output"]["data_exports_dir"]]:
        os.makedirs(d, exist_ok=True)

    # Load integrated dataset
    print("Loading dataset...")
    df = load_combined_dataset(cfg)
    if df.empty:
        print("ERROR: No data loaded.")
        sys.exit(1)

    top_clusters = get_top_clusters(df, analysis["top_n_clusters"])
    top_subs = get_top_subreddits(df, analysis["top_m_subreddits"])
    print(f"Top clusters: {top_clusters[:5]}...")
    print(f"Top subreddits: {top_subs[:5]}...")

    # Train models
    print("\n--- Training cluster models ---")
    cluster_models = train_all_cluster_models(df, top_clusters, cfg)

    print("\n--- Training subreddit models ---")
    sub_models = train_all_subreddit_models(df, top_subs, cfg)

    # Analysis outputs
    viz_dir = cfg["output"]["visualizations_dir"]
    export_dir = cfg["output"]["data_exports_dir"]

    # Topic overlap heatmap
    if len(sub_models) >= 2:
        print("\nComputing topic overlap matrix...")
        overlap = topic_overlap_matrix(
            list(sub_models.keys()), cfg["output"]["topic_models_dir"],
            threshold=analysis.get("similarity_threshold", 0.3)
        )
        overlap.to_csv(os.path.join(export_dir, "topic_overlap_matrix.csv"))
        plot_overlap_heatmap(overlap, os.path.join(viz_dir, "topic_overlap_heatmap.png"))

    # Cluster distribution
    if sub_models and top_clusters:
        matrix = get_cluster_subreddit_matrix(df, top_clusters, list(sub_models.keys()))
        matrix.to_csv(os.path.join(export_dir, "cluster_subreddit_distribution.csv"))
        plot_cluster_distribution(matrix, os.path.join(viz_dir, "cluster_distribution.png"))

    # Diversity scores
    div_data = [{"cluster_id": c, "diversity": cluster_diversity_score(df, c)}
                for c in top_clusters]
    pd.DataFrame(div_data).to_csv(os.path.join(export_dir, "cluster_diversity.csv"), index=False)
    plot_diversity_scores(df, analysis["top_n_clusters"],
                          os.path.join(viz_dir, "diversity_scores.png"))

    # Summary statistics
    total_c = sum(m["n_topics"] for m in cluster_models.values())
    total_s = sum(m["n_topics"] for m in sub_models.values())
    stats = {
        "total_documents": len(df),
        "unique_clusters": len([c for c in df["cluster_id"].unique() if c != -1]),
        "unique_subreddits": df["subreddit"].nunique(),
        "clusters_modeled": len(cluster_models),
        "subreddits_modeled": len(sub_models),
        "topics_from_clusters": total_c,
        "topics_from_subreddits": total_s,
        "total_topics": total_c + total_s,
        "noise_points": (df["cluster_id"] == -1).sum(),
    }
    pd.DataFrame([stats]).to_csv(os.path.join(export_dir, "summary_statistics.csv"), index=False)

    print(f"\nDone. {total_c + total_s} topics discovered "
          f"({len(cluster_models)} cluster + {len(sub_models)} subreddit models).")
    print(f"Results: {cfg['output']['results_dir']}")


if __name__ == "__main__":
    main()

