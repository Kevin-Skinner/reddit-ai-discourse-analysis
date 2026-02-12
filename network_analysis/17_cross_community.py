"""
Stage 17: Cross-Community Topic Analysis
==========================================
We analyze how topics connect communities by examining the similarity edge
network from Stage 15. Three main analyses:

1. **Cross-scope connections**: Which cluster topics align with which subreddit
   topics? High-similarity cluster-subreddit edges reveal how the data-driven
   clusters (HDBSCAN) relate to the community-defined groups (subreddits).

2. **Bridge topics**: Topics whose centroids are similar to topics in 3+ different
   subreddits. These are the shared concerns that span multiple communities --
   e.g., AI safety, job displacement, model capabilities.

3. **Topic reach**: For each topic, how many distinct communities does it "reach"
   via high-similarity edges? Topics with high reach are the most universal;
   topics with low reach are community-specific.

4. **Community affinity network**: A NetworkX graph of subreddits connected by
   shared topic similarity, visualized as a force-directed layout.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import yaml
import warnings
warnings.filterwarnings("ignore")


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data(cfg):
    export_dir = cfg["output"]["data_exports_dir"]
    df_edges = pd.read_parquet(os.path.join(export_dir, "topic_similarity_edges.parquet"))
    df_centroids = pd.read_parquet(os.path.join(export_dir, "topic_centroids_3d.parquet"))
    print(f"Loaded {len(df_edges):,} edges, {len(df_centroids)} centroids")
    return df_edges, df_centroids


# ---- 1. Cross-scope connections ----

def analyze_cross_scope(df_edges, cross_threshold=0.6):
    """Find high-similarity connections between cluster and subreddit topics."""
    cross = df_edges[df_edges["source_scope"] != df_edges["target_scope"]].copy()
    high = cross[cross["similarity"] >= cross_threshold]

    rows = []
    for _, r in high.iterrows():
        if r["source_scope"] == "cluster":
            rows.append({"cluster": r["source_value"], "subreddit": r["target_value"],
                         "similarity": r["similarity"]})
        else:
            rows.append({"cluster": r["target_value"], "subreddit": r["source_value"],
                         "similarity": r["similarity"]})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = df.groupby(["cluster", "subreddit"]).agg(
        n_connections=("similarity", "count"),
        avg_similarity=("similarity", "mean"),
        max_similarity=("similarity", "max"),
    ).reset_index().sort_values("n_connections", ascending=False)
    return agg


# ---- 2. Bridge topics ----

def find_bridge_topics(df_edges, min_communities=3, high_sim=0.7):
    """Identify topics that bridge 3+ subreddits."""
    sub2sub = df_edges[
        (df_edges["source_scope"] == "subreddit") &
        (df_edges["target_scope"] == "subreddit") &
        (df_edges["similarity"] >= high_sim)
    ]
    connections = defaultdict(set)
    for _, r in sub2sub.iterrows():
        sk = f"{r['source_value']}_topic{r['source_topic']}"
        tk = f"{r['target_value']}_topic{r['target_topic']}"
        connections[sk].add(r["target_value"])
        connections[tk].add(r["source_value"])

    rows = []
    for key, linked in connections.items():
        parts = key.rsplit("_topic", 1)
        sub, tid = parts[0], int(parts[1]) if len(parts) > 1 else 0
        all_subs = linked | {sub}
        if len(all_subs) >= min_communities:
            rows.append({
                "subreddit": sub, "topic_id": tid,
                "n_communities": len(all_subs),
                "connected_subreddits": ", ".join(sorted(all_subs))
            })
    df = pd.DataFrame(rows)
    return df.sort_values("n_communities", ascending=False) if not df.empty else df


# ---- 3. Topic reach ----

def compute_topic_reach(df_edges, high_sim=0.7):
    """Count how many distinct communities each topic reaches."""
    sig = df_edges[df_edges["similarity"] >= high_sim]
    reach = defaultdict(set)
    for _, r in sig.iterrows():
        sk = (r["source_scope"], str(r["source_value"]), int(r["source_topic"]))
        tk = (r["target_scope"], str(r["target_value"]), int(r["target_topic"]))
        reach[sk].add(f"{r['target_scope']}:{r['target_value']}")
        reach[tk].add(f"{r['source_scope']}:{r['source_value']}")

    rows = [{"scope_type": k[0], "scope_value": k[1], "topic_id": k[2],
             "reach": len(v), "communities": ", ".join(sorted(list(v)[:10]))}
            for k, v in reach.items()]
    df = pd.DataFrame(rows)
    return df.sort_values("reach", ascending=False) if not df.empty else df


# ---- 4. Affinity network ----

def build_affinity_network(df_edges, high_sim=0.7, min_connections=5, out_path=None):
    """Build and visualize a subreddit affinity network."""
    sub_edges = df_edges[
        (df_edges["source_scope"] == "subreddit") &
        (df_edges["target_scope"] == "subreddit") &
        (df_edges["similarity"] >= high_sim)
    ]
    pairs = sub_edges.groupby(["source_value", "target_value"]).agg(
        n=("similarity", "count"), avg_sim=("similarity", "mean")
    ).reset_index()
    pairs = pairs[pairs["n"] >= min_connections]

    if pairs.empty:
        return None

    G = nx.Graph()
    for _, r in pairs.iterrows():
        G.add_edge(r["source_value"], r["target_value"],
                    weight=r["n"], avg_sim=r["avg_sim"])
    G.remove_nodes_from(list(nx.isolates(G)))

    if out_path:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        degrees = dict(G.degree())
        sizes = [degrees[n] * 100 + 100 for n in G.nodes()]

        plt.figure(figsize=(16, 12))
        nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color="gray")
        nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="steelblue", alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
        plt.title("Subreddit Topic Affinity Network", fontsize=16, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved network plot: {out_path}")

    return G


# ---- Main ----

def main():
    cfg = load_config()
    export_dir = cfg["output"]["data_exports_dir"]
    viz_dir = cfg["output"]["visualizations_dir"]
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    df_edges, df_centroids = load_data(cfg)

    # Cross-scope
    df_cross = analyze_cross_scope(df_edges)
    if not df_cross.empty:
        df_cross.to_csv(os.path.join(export_dir, "cross_community_topics.csv"), index=False)
        print(f"  Saved {len(df_cross)} cross-scope connections")

    # Bridge topics
    df_bridges = find_bridge_topics(df_edges)
    if not df_bridges.empty:
        df_bridges.to_csv(os.path.join(export_dir, "bridge_topics_ranked.csv"), index=False)
        print(f"  Found {len(df_bridges)} bridge topics")

    # Reach
    df_reach = compute_topic_reach(df_edges)
    if not df_reach.empty:
        df_reach.to_csv(os.path.join(export_dir, "topic_reach_scores.csv"), index=False)
        print(f"  Computed reach for {len(df_reach)} topics")

    # Affinity network
    G = build_affinity_network(
        df_edges, out_path=os.path.join(viz_dir, "topic_affinity_network.png")
    )
    if G:
        rows = [{"source": u, "target": v, "n": d["weight"], "avg_sim": d["avg_sim"]}
                for u, v, d in G.edges(data=True)]
        pd.DataFrame(rows).to_csv(
            os.path.join(export_dir, "community_affinity_network.csv"), index=False
        )


if __name__ == "__main__":
    main()

