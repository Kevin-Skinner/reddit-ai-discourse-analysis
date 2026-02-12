"""
Stage 10: Interactive 3D Cluster Visualization
================================================
We build an interactive PyVista point cloud of the clustered 3D embeddings.
The visualization supports two coloring modes:

1. **Cluster mode** (press 'c'): Points colored by their HDBSCAN cluster ID
2. **Subreddit mode** (press 's'): Points colored by subreddit frequency rank,
   with a slider to control how many top subreddits are visible

This interactive view is invaluable for qualitative exploration -- we can rotate
the point cloud, zoom into dense regions, and observe how subreddit communities
map onto the semantic embedding space.
"""

import os
import duckdb
import pandas as pd
import numpy as np
import pyvista as pv
import yaml


pv.set_jupyter_backend(None)


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_subreddits_for_ids(db_path, ids):
    """Look up subreddit names for a list of document IDs."""
    if not ids:
        return pd.DataFrame()
    try:
        con = duckdb.connect(db_path, read_only=True)
        meta = con.execute(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_type='BASE TABLE' LIMIT 1"
        ).fetchone()
        tbl = f"{meta[0]}.{meta[1]}"
        id_str = "'" + "','".join(ids) + "'"
        result = con.execute(f"SELECT id, subreddit FROM {tbl} WHERE id IN ({id_str})").fetchdf()
        con.close()
        return result
    except Exception:
        return pd.DataFrame()


def run_viz(cfg):
    sample_size = cfg.get("animation", {}).get("sample_size", 200_000)

    # Load 3D coordinates
    con = duckdb.connect(cfg["duckdb"]["embeddings_3d_db"], read_only=True)
    df_coords = con.execute("SELECT id, type, x, y, z FROM embeddings_3d").fetchdf()
    con.close()

    # Load cluster assignments
    assignments_path = os.path.join(cfg["output"]["results_dir"], "cluster_assignments.parquet")
    df_clusters = pd.read_parquet(assignments_path)
    df = pd.merge(df_coords, df_clusters, on="id")

    # Sample for performance
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    # Look up subreddit names from the source databases
    ids_posts = df[df["type"] == "post"]["id"].tolist()
    ids_comments = df[df["type"] == "comment"]["id"].tolist()

    df_subs_p = get_subreddits_for_ids(cfg["duckdb"]["tokenized_posts_db"], ids_posts)
    df_subs_c = get_subreddits_for_ids(cfg["duckdb"]["tokenized_comments_db"], ids_comments)
    df_subs = pd.concat([df_subs_p, df_subs_c])

    df = pd.merge(df, df_subs, on="id", how="left")
    df["subreddit"] = df["subreddit"].fillna("Unknown")

    # Build subreddit frequency ranking
    sub_counts = df["subreddit"].value_counts()
    sub_counts = sub_counts.drop(labels=["Unknown"], errors="ignore")
    ordered_subs = sub_counts.index.tolist()
    rank_map = {sub: i + 1 for i, sub in enumerate(ordered_subs)}

    df["sub_rank"] = df["subreddit"].map(rank_map).fillna(999_999)
    df["cluster_int"] = df["cluster_id"].astype(int)

    # Build PyVista point cloud
    points = pv.PolyData(df[["x", "y", "z"]].values)
    points["cluster_id"] = df["cluster_int"].values
    points["sub_rank"] = df["sub_rank"].values.astype(np.float32)

    plotter = pv.Plotter()
    plotter.set_background("black")
    plotter.add_mesh(
        points, scalars="cluster_id", cmap="tab20", point_size=3,
        render_points_as_spheres=True, show_scalar_bar=False,
        opacity=0.6, name="main_cloud"
    )

    # -- Interactive controls --

    def update_sidebar(n_subs, visible=True):
        """Show a ranked leaderboard of visible subreddits."""
        for i in range(12):
            plotter.remove_actor(f"col{i}")
        if not visible or not ordered_subs:
            return
        current_list = ordered_subs[:int(n_subs)]
        chunk_size = 10
        cols = [current_list[i:i + chunk_size] for i in range(0, len(current_list), chunk_size)]
        for idx, col_data in enumerate(cols):
            rank_start = 1 + (idx * chunk_size)
            lines = [f"{rank_start + i}. {name}" for i, name in enumerate(col_data)]
            plotter.add_text(
                "\n".join(lines),
                position=(0.01 + idx * 0.098, 0.85),
                font_size=9, color="white", name=f"col{idx}", viewport=True
            )

    def switch_to_clusters():
        plotter.add_mesh(
            points, scalars="cluster_id", cmap="tab20", point_size=3,
            render_points_as_spheres=True, show_scalar_bar=False, name="main_cloud"
        )
        plotter.add_text("Mode: Clusters", name="status", font_size=12,
                         position="lower_left", color="white")
        update_sidebar(0, visible=False)

    def filter_subs(value):
        value = int(value)
        thresh = points.threshold([1, value], scalars="sub_rank", preference="cell")
        plotter.add_mesh(
            thresh, scalars="sub_rank", cmap="turbo", point_size=3,
            render_points_as_spheres=True, show_scalar_bar=False, name="main_cloud"
        )
        plotter.add_text(
            f"Showing Top {value} Subreddits", name="status",
            font_size=12, position="lower_left", color="white"
        )
        update_sidebar(value, visible=True)

    plotter.add_key_event("c", switch_to_clusters)
    plotter.add_key_event("s", lambda: filter_subs(10))
    plotter.add_slider_widget(
        filter_subs, [1, 100], title="Top N Subs",
        pointa=(0.05, 0.1), pointb=(0.25, 0.1), style="modern"
    )
    plotter.add_text("Press 'c' for Clusters, 's' for Subreddits",
                     position="lower_left", font_size=10)
    plotter.show()


def main():
    cfg = load_config()
    run_viz(cfg)


if __name__ == "__main__":
    main()

