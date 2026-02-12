"""
Generate a rotating 3D GIF of the clustered embedding space.

Points appear progressively by subreddit popularity rank (most-active first)
while the camera orbits the cloud, then the process reverses — producing a
"bloom-and-fade" effect that highlights community structure.

Requires:
    - 3D UMAP coordinates  (DuckDB table `embeddings_3d`)
    - HDBSCAN cluster assignments  (Parquet)
    - Tokenised posts / comments DBs  (for subreddit labels)

Output:
    - An animated GIF written to `assets/cluster_evolution.gif`
"""

import yaml
import duckdb
import numpy as np
import pandas as pd
import pyvista as pv

# ── Configuration ────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

COORDS_DB        = cfg["duckdb"]["embeddings_3d_db"]
ASSIGNMENTS_PATH = cfg["output"]["results_dir"] + "/cluster_assignments.parquet"
POSTS_DB         = cfg["duckdb"]["tokenized_posts_db"]
COMMENTS_DB      = cfg["duckdb"]["tokenized_comments_db"]
OUTPUT_GIF       = cfg["animation"]["output_file"]

SAMPLE_SIZE  = cfg["animation"].get("sample_size", 200_000)
FPS          = cfg["animation"].get("fps", 15)
DURATION_SEC = cfg["animation"].get("duration_sec", 50)
TOTAL_FRAMES = FPS * DURATION_SEC

# ── Helpers ──────────────────────────────────────────────────────────

def _fetch_subreddits(db_path: str, ids: list[str]) -> pd.DataFrame:
    """Look up subreddit names for a list of post/comment IDs."""
    if not ids:
        return pd.DataFrame(columns=["id", "subreddit"])
    con = duckdb.connect(db_path, read_only=True)
    meta = con.execute(
        "SELECT table_schema, table_name "
        "FROM information_schema.tables "
        "WHERE table_type='BASE TABLE' LIMIT 1"
    ).fetchone()
    table = f"{meta[0]}.{meta[1]}"
    id_csv = ",".join(f"'{i}'" for i in ids)
    result = con.execute(
        f"SELECT id, subreddit FROM {table} WHERE id IN ({id_csv})"
    ).fetchdf()
    con.close()
    return result


def _build_dataframe() -> tuple[pd.DataFrame, list[str]]:
    """Load coordinates, cluster labels, and subreddit names; return
    the merged DataFrame and an ordered list of subreddits by frequency."""

    # 3D coordinates
    con = duckdb.connect(COORDS_DB, read_only=True)
    coords = con.execute("SELECT id, type, x, y, z FROM embeddings_3d").fetchdf()
    con.close()

    # Cluster assignments
    clusters = pd.read_parquet(ASSIGNMENTS_PATH)
    df = coords.merge(clusters, on="id")

    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42)

    # Subreddit labels (posts + comments)
    post_ids    = df.loc[df["type"] == "post",    "id"].tolist()
    comment_ids = df.loc[df["type"] == "comment", "id"].tolist()
    subs = pd.concat([
        _fetch_subreddits(POSTS_DB, post_ids),
        _fetch_subreddits(COMMENTS_DB, comment_ids),
    ])
    df = df.merge(subs, on="id", how="left")
    df["subreddit"] = df["subreddit"].fillna("Unknown")

    # Popularity ranking (1 = most active)
    counts = df["subreddit"].value_counts().drop(labels=["Unknown", None], errors="ignore")
    ordered = counts.index.tolist()
    rank_map = {sub: i + 1 for i, sub in enumerate(ordered)}

    df["sub_rank"]    = df["subreddit"].map(rank_map).fillna(999_999).astype(np.float32)
    df["cluster_int"] = df["cluster_id"].astype(int)

    return df, ordered


# ── Main ─────────────────────────────────────────────────────────────

def generate_gif():
    pv.set_jupyter_backend(None)  # force desktop/off-screen backend

    print("Loading data & computing subreddit ranks …")
    df, ordered_subs = _build_dataframe()

    cloud = pv.PolyData(df[["x", "y", "z"]].values)
    cloud["cluster_id"] = df["cluster_int"].values
    cloud["sub_rank"]   = df["sub_rank"].values

    # ── Scene setup ──────────────────────────────────────────────────
    plotter = pv.Plotter(off_screen=True, window_size=[1280, 720])
    plotter.set_background("black")

    plotter.add_mesh(
        cloud.threshold([1, 1], scalars="sub_rank"),
        scalars="sub_rank", cmap="turbo", clim=[1, 100],
        point_size=3, render_points_as_spheres=True,
        show_scalar_bar=False, name="main_cloud",
    )

    # Camera
    plotter.camera_position = "xy"
    plotter.camera.azimuth   = 50
    plotter.camera.elevation = 50
    plotter.camera.zoom(1.4)

    focus = list(plotter.camera.focal_point)
    focus[1] += 2.0                         # shift cloud downward in frame
    plotter.camera.focal_point = focus
    plotter.camera.zoom(1.4)

    # ── Leaderboard overlay ──────────────────────────────────────────
    def _draw_leaderboard(n: int):
        for i in range(12):
            plotter.remove_actor(f"col{i}")

        visible = ordered_subs[: int(n)]
        chunk = 10
        cols  = [visible[i : i + chunk] for i in range(0, len(visible), chunk)]

        for idx, col_data in enumerate(cols):
            lines = "\n".join(
                f"{1 + idx * chunk + j}. {name}" for j, name in enumerate(col_data)
            )
            plotter.add_text(
                lines,
                position=(0.01 + idx * 0.098, 0.68),
                font_size=8, color="white",
                name=f"col{idx}", viewport=True,
            )

    # ── Render loop ──────────────────────────────────────────────────
    print(f"Rendering {TOTAL_FRAMES} frames → {OUTPUT_GIF} …")
    plotter.open_gif(OUTPUT_GIF, fps=FPS)

    halfway = TOTAL_FRAMES // 2
    for frame in range(TOTAL_FRAMES):
        # Ramp up to 100 subs, then back down
        if frame <= halfway:
            n_subs = 1 + 99 * (frame / halfway)
        else:
            n_subs = 100 - 99 * ((frame - halfway) / halfway)
        n_subs = max(1, min(100, int(n_subs)))

        # Update point cloud threshold
        thresh = cloud.threshold([1, n_subs], scalars="sub_rank", preference="cell")
        plotter.add_mesh(
            thresh, scalars="sub_rank", cmap="turbo", clim=[1, 100],
            point_size=3, render_points_as_spheres=True,
            show_scalar_bar=False, name="main_cloud",
        )

        _draw_leaderboard(n_subs)
        plotter.camera.azimuth += 0.96      # ~2 full rotations over 750 frames
        plotter.write_frame()

        if frame % 50 == 0:
            print(f"  frame {frame}/{TOTAL_FRAMES}  (visible subs = {n_subs})")

    plotter.close()
    print(f"Done → {OUTPUT_GIF}")


if __name__ == "__main__":
    generate_gif()
