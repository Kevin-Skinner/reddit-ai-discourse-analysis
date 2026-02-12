"""
Stage 18: 3D Subreddit Similarity Network
===========================================
We build an interactive 3D Plotly network where each node is a subreddit,
positioned by a force-directed layout based on pairwise topic similarity,
and colored by its dominant HDBSCAN cluster. This visualization answers:
"Which subreddits talk about similar AI topics, and are they in the same
semantic clusters?"

Key features:
- Node color = dominant HDBSCAN cluster (from the cluster distribution analysis)
- Node size = connection degree (how many other subreddits share topics)
- Hover text shows: dominant cluster, cluster doc proportion, total topics,
  and top-10 most similar subreddits ranked by topic overlap
- Topic overlap is capped by the neighbor's topic count to avoid inflated scores
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import warnings
warnings.filterwarnings("ignore")


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_dominant_clusters(df_dist):
    """Find each subreddit's dominant cluster from the distribution matrix."""
    sub_cols = [c for c in df_dist.columns if c != "cluster_id"]
    dom_clusters, dom_counts, totals = {}, {}, {}
    for sub in sub_cols:
        idx = df_dist[sub].idxmax()
        dom_clusters[sub] = df_dist.loc[idx, "cluster_id"]
        dom_counts[sub] = int(df_dist.loc[idx, sub])
        totals[sub] = int(df_dist[sub].sum())
    return dom_clusters, dom_counts, totals


def compute_topic_overlaps(subreddits, centroids_path, threshold=0.7):
    """
    For each (A, B) pair, count how many of A's topic centroids have at least
    one match in B with cosine similarity >= threshold.
    """
    df = pd.read_parquet(centroids_path)
    df = df[df["scope_value"].isin(subreddits)]

    sub_vecs = {}
    for sub in subreddits:
        sdf = df[df["scope_value"] == sub]
        if len(sdf) > 0:
            sub_vecs[sub] = np.vstack(sdf["centroid"].values)

    overlaps = {}
    subs = list(sub_vecs.keys())
    for a in subs:
        for b in subs:
            if a == b:
                continue
            sim = cosine_similarity(sub_vecs[a], sub_vecs[b])
            overlaps[(a, b)] = int((sim.max(axis=1) >= threshold).sum())
    return overlaps


def build_network(df_similarity, min_score=0.1):
    """Build a NetworkX graph from a similarity-pairs CSV."""
    df = df_similarity[df_similarity["combined_score"] >= min_score]
    G = nx.Graph()
    for _, r in df.iterrows():
        G.add_edge(r["subreddit_1"], r["subreddit_2"],
                    weight=r["combined_score"],
                    shared_topics=int(r.get("shared_topics", 0)),
                    shared_clusters=int(r.get("shared_clusters", 0)))
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def create_3d_network(G, dom_clusters, dom_counts, totals,
                       topic_counts, overlaps, out_path):
    """Render the 3D Plotly network visualization."""
    palette = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        "#F8B500", "#00CED1", "#FF69B4", "#32CD32", "#FFD700",
        "#8A2BE2", "#00FA9A", "#DC143C", "#00BFFF", "#FF4500",
    ]
    unique_c = sorted(set(dom_clusters.values()))
    c2color = {c: palette[i % len(palette)] for i, c in enumerate(unique_c)}

    pos = nx.spring_layout(G, dim=3, k=2, iterations=100, seed=42, weight="weight")
    nodes = list(G.nodes())
    degrees = dict(G.degree())

    def capped(a, b):
        raw = overlaps.get((a, b), 0)
        cap = topic_counts.get(b, 0)
        return min(raw, cap)

    hover = []
    for n in nodes:
        t_total = topic_counts.get(n, 0)
        d_count = dom_counts.get(n, 0)
        t_count = totals.get(n, 1)
        pct = 100 * d_count / max(t_count, 1)
        neighbors = sorted(G.neighbors(n), key=lambda nb: capped(n, nb) / max(t_total, 1), reverse=True)[:10]

        txt = (f"<b>r/{n}</b><br>Dominant Cluster: {dom_clusters.get(n, '?')}<br>"
               f"Cluster Docs: {d_count:,}/{t_count:,} ({pct:.1f}%)<br>"
               f"Topics: {t_total}<br>Connections: {degrees[n]}<br><br>"
               f"<b>Most similar:</b><br>")
        for nb in neighbors:
            ov = capped(n, nb)
            ov_pct = 100 * ov / max(t_total, 1)
            sc = G[n][nb].get("shared_clusters", 0)
            txt += f"  r/{nb}: {ov}/{t_total} topics ({ov_pct:.0f}%), {sc} clusters<br>"
        hover.append(txt)

    fig = go.Figure(go.Scatter3d(
        x=[pos[n][0] for n in nodes],
        y=[pos[n][1] for n in nodes],
        z=[pos[n][2] for n in nodes],
        mode="markers+text",
        marker=dict(
            size=[degrees[n] * 0.5 + 5 for n in nodes],
            color=[c2color.get(dom_clusters.get(n), "#888") for n in nodes],
            opacity=0.9, line=dict(width=1, color="white"),
        ),
        text=[f"r/{n}" for n in nodes],
        textposition="top center", textfont=dict(size=9, color="white"),
        hovertext=hover, hoverinfo="text", name="Subreddits"
    ))

    fig.update_layout(
        title=dict(
            text="<b>3D Subreddit Similarity Network</b><br>"
                 "<sup>Colored by dominant HDBSCAN cluster</sup>",
            font=dict(size=20, color="white"), x=0.5
        ),
        showlegend=False, hovermode="closest",
        paper_bgcolor="rgb(20,20,30)", plot_bgcolor="rgb(20,20,30)",
        width=1400, height=900,
        scene=dict(
            xaxis=dict(showbackground=True, backgroundcolor="rgb(30,30,45)",
                       showgrid=True, gridcolor="rgb(50,50,70)",
                       showticklabels=False, title=""),
            yaxis=dict(showbackground=True, backgroundcolor="rgb(30,30,45)",
                       showgrid=True, gridcolor="rgb(50,50,70)",
                       showticklabels=False, title=""),
            zaxis=dict(showbackground=True, backgroundcolor="rgb(30,30,45)",
                       showgrid=True, gridcolor="rgb(50,50,70)",
                       showticklabels=False, title=""),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.write_html(out_path)
    print(f"  Saved: {out_path}")


def main():
    cfg = load_config()
    export_dir = cfg["output"]["data_exports_dir"]
    viz_dir = cfg["output"]["visualizations_dir"]

    # Load cluster distribution
    dist_path = os.path.join(export_dir, "cluster_subreddit_distribution.csv")
    df_dist = pd.read_csv(dist_path)
    dom_clusters, dom_counts, totals = get_dominant_clusters(df_dist)

    # Load similarity pairs (produced by comparison analysis or a separate step)
    sim_path = os.path.join(export_dir, "combined_similarity.csv")
    if not os.path.exists(sim_path):
        print(f"  Similarity file not found at {sim_path}; skipping network.")
        return
    df_sim = pd.read_csv(sim_path)

    G = build_network(df_sim)
    network_subs = set(G.nodes())

    # Topic overlaps
    centroids_path = os.path.join(export_dir, "topic_centroids_subreddits.parquet")
    overlaps = compute_topic_overlaps(network_subs, centroids_path)

    # Dummy topic counts (could load from model summaries)
    topic_counts = {s: 0 for s in network_subs}
    try:
        df_centroids = pd.read_parquet(centroids_path)
        for s in network_subs:
            topic_counts[s] = len(df_centroids[df_centroids["scope_value"] == s])
    except Exception:
        pass

    out = os.path.join(viz_dir, "similarity_network_3d.html")
    create_3d_network(G, dom_clusters, dom_counts, totals,
                       topic_counts, overlaps, out)


if __name__ == "__main__":
    main()

