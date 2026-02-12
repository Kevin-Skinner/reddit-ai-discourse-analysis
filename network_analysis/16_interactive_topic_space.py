"""
Stage 16: Interactive 3D Topic Space Visualization
====================================================
We create an interactive Plotly visualization of the topic centroid space.
Each point is a topic (from either a cluster model or a subreddit model),
positioned by its UMAP-reduced 3D coordinates (from Stage 15).

Features:
- Points colored by scope type (cluster vs subreddit) or by community
- Point size scaled by document count (log scale)
- Similarity edges shown as translucent lines between related topics
- Rich hover text with scope info, topic keywords, and document count
- Subreddit-only filtered view for focused exploration

Output: Interactive HTML files that can be opened in any browser.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
import warnings
warnings.filterwarnings("ignore")


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data(cfg):
    export_dir = cfg["output"]["data_exports_dir"]
    df_centroids = pd.read_parquet(os.path.join(export_dir, "topic_centroids_3d.parquet"))
    df_edges = pd.read_parquet(os.path.join(export_dir, "topic_similarity_edges.parquet"))
    print(f"Loaded {len(df_centroids)} centroids, {len(df_edges):,} edges")
    return df_centroids, df_edges


def filter_edges(df_edges, min_sim=0.85, max_n=5000):
    """Keep only the strongest edges for rendering performance."""
    df = df_edges[df_edges["similarity"] >= min_sim].copy()
    if len(df) > max_n:
        df = df.nlargest(max_n, "similarity")
    print(f"  Rendering {len(df)} edges (threshold={min_sim})")
    return df


def edge_trace(df_centroids, df_edges):
    """Build a single Scatter3d trace for all edges."""
    lookup = {}
    for _, r in df_centroids.iterrows():
        key = f"{r['scope_type']}_{r['scope_value']}_{r['topic_id']}"
        lookup[key] = (r["x"], r["y"], r["z"])

    xs, ys, zs = [], [], []
    for _, e in df_edges.iterrows():
        sk = f"{e['source_scope']}_{e['source_value']}_{e['source_topic']}"
        tk = f"{e['target_scope']}_{e['target_value']}_{e['target_topic']}"
        if sk in lookup and tk in lookup:
            s, t = lookup[sk], lookup[tk]
            xs += [s[0], t[0], None]
            ys += [s[1], t[1], None]
            zs += [s[2], t[2], None]

    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color="rgba(150,150,150,0.15)", width=0.5),
        hoverinfo="none", name="Similarity Edges"
    )


def node_trace(df):
    """Scatter3d for topic centroids, colored by scope type."""
    colors = df["scope_type"].map({"cluster": 0, "subreddit": 1})
    sizes = np.clip(np.log1p(df["n_docs"]) * 2 + 3, 3, 15)

    hover = [
        f"<b>{r['scope_type'].title()}: {r['scope_value']}</b><br>"
        f"Topic {r['topic_id']}<br>Docs: {int(r['n_docs']):,}"
        for _, r in df.iterrows()
    ]

    return go.Scatter3d(
        x=df["x"], y=df["y"], z=df["z"], mode="markers",
        marker=dict(
            size=sizes, color=colors,
            colorscale=[[0, "#FF6B6B"], [1, "#4ECDC4"]],
            opacity=0.7, line=dict(width=0.5, color="white"),
        ),
        text=hover, hoverinfo="text", name="Topic Centroids"
    )


def build_main_viz(df_centroids, df_edges, out_path):
    """Build and save the full interactive visualization."""
    fig = go.Figure()

    if not df_edges.empty:
        fig.add_trace(edge_trace(df_centroids, df_edges))
    fig.add_trace(node_trace(df_centroids))

    fig.update_layout(
        title=dict(
            text="<b>Topic Space: Reddit AI Discourse</b><br>"
                 "<sub>Interactive 3D visualization of BERTopic centroids</sub>",
            x=0.5, font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(title="UMAP-1"), yaxis=dict(title="UMAP-2"),
            zaxis=dict(title="UMAP-3"),
            bgcolor="rgba(250,250,250,1)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=80),
        showlegend=True, paper_bgcolor="white", height=800,
        annotations=[dict(
            text="Red = cluster topics | Teal = subreddit topics | Size = doc count",
            showarrow=False, x=0.5, y=-0.05, xref="paper", yref="paper",
            font=dict(size=12, color="gray")
        )]
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)
    print(f"  Saved: {out_path}")


def build_subreddit_viz(df_centroids, out_path):
    """Subreddit-only view colored by community."""
    df = df_centroids[df_centroids["scope_type"] == "subreddit"].copy()
    subs = sorted(df["scope_value"].unique())
    cmap = {s: i for i, s in enumerate(subs)}
    colors = df["scope_value"].map(cmap)
    sizes = np.clip(np.log1p(df["n_docs"]) * 2 + 4, 4, 18)

    hover = [
        f"<b>r/{r['scope_value']}</b><br>Topic {r['topic_id']}<br>Docs: {int(r['n_docs']):,}"
        for _, r in df.iterrows()
    ]

    fig = go.Figure(go.Scatter3d(
        x=df["x"], y=df["y"], z=df["z"], mode="markers",
        marker=dict(size=sizes, color=colors, colorscale="Turbo",
                    opacity=0.75, line=dict(width=0.3, color="white")),
        text=hover, hoverinfo="text", name="Subreddit Topics"
    ))
    fig.update_layout(
        title=dict(text="<b>Subreddit Topic Space</b>", x=0.5, font=dict(size=18)),
        scene=dict(xaxis=dict(title="UMAP-1"), yaxis=dict(title="UMAP-2"),
                   zaxis=dict(title="UMAP-3"), bgcolor="rgba(250,250,250,1)"),
        margin=dict(l=0, r=0, b=0, t=80), paper_bgcolor="white", height=800
    )
    sub_path = out_path.replace(".html", "_subreddits.html")
    fig.write_html(sub_path, include_plotlyjs=True, full_html=True)
    print(f"  Saved: {sub_path}")


def main():
    cfg = load_config()
    viz_dir = cfg["output"]["visualizations_dir"]
    os.makedirs(viz_dir, exist_ok=True)

    df_centroids, df_edges = load_data(cfg)
    df_edges_f = filter_edges(df_edges)

    out = os.path.join(viz_dir, "topic_3d_interactive.html")
    build_main_viz(df_centroids, df_edges_f, out)
    build_subreddit_viz(df_centroids, out)


if __name__ == "__main__":
    main()

