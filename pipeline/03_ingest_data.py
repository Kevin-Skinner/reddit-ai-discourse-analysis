"""
Stage 03: Ingest Arctic Shift Data
====================================
We read compressed .zst files from the Arctic Shift Reddit dumps and insert
matching records into DuckDB. The key insight is that DuckDB can read
newline-delimited JSON directly (including from compressed files), so we
leverage a SQL INSERT...SELECT with a JOIN against our valid_subs table to
filter at read time rather than loading everything into memory first.

Usage:
    python 03_ingest_data.py --mode comments
    python 03_ingest_data.py --mode posts
"""

import argparse
import re
import datetime
import os
from pathlib import Path
import yaml
import duckdb


# Patterns to match Arctic Shift filenames: RC_YYYY-MM.zst (comments), RS_YYYY-MM.zst (posts)
RC_PAT = re.compile(r"RC_(\d{4})-(\d{2})\.zst$", re.IGNORECASE)
RS_PAT = re.compile(r"RS_(\d{4})-(\d{2})\.zst$", re.IGNORECASE)


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def discover_files(cfg, mode):
    """Walk data_roots to find .zst files matching the target years."""
    pattern = RC_PAT if mode == "comments" else RS_PAT
    years = set(cfg["data"].get("years", []))
    files = []

    for root in cfg["data"].get("data_roots", []):
        root_path = Path(root)
        if not root_path.exists():
            continue
        for p in root_path.rglob("*.zst"):
            m = pattern.match(p.name)
            if m and int(m.group(1)) in years:
                files.append(p)

    return sorted(files)


def upsert_state(con, schema, table, filepath, status):
    """Track ingestion state for resumability."""
    now = datetime.datetime.now(datetime.UTC).isoformat()
    con.execute(f"""
        INSERT INTO {schema}.{table}(filepath, last_line, status, updated_at)
        VALUES (?, 0, ?, ?)
        ON CONFLICT (filepath) DO UPDATE
        SET status=EXCLUDED.status, updated_at=EXCLUDED.updated_at
    """, [str(filepath), status, now])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["comments", "posts"], required=True)
    args = parser.parse_args()

    cfg = load_config()
    schema = cfg["duckdb"].get("schema", "reddit")
    state_table = f"{args.mode}_state"

    db_path = cfg["duckdb"][f"{args.mode}_db"]
    con = duckdb.connect(db_path)

    # Performance tuning
    threads = max(2, (os.cpu_count() or 4) - 1)
    con.execute(f"PRAGMA threads={threads}")
    con.execute("PRAGMA memory_limit='12GB'")

    files = discover_files(cfg, args.mode)
    print(f"Found {len(files)} {args.mode} files to ingest")

    for f in files:
        upsert_state(con, schema, state_table, f, "in_progress")

        # Build the INSERT...SELECT that reads JSON directly and joins against valid_subs.
        # DuckDB's read_json_auto handles .zst decompression transparently.
        if args.mode == "comments":
            target = f"{schema}.filtered_comments"
            select_cols = """
                CAST(t.id AS TEXT), CAST(t.subreddit AS TEXT), CAST(t.body AS TEXT),
                CAST(t.link_id AS TEXT), CAST(t.parent_id AS TEXT),
                CAST(t.created_utc AS BIGINT), CAST(t.score AS BIGINT),
                CAST(t.author AS TEXT), CAST(t.subreddit_id AS TEXT),
                CAST(t.retrieved_on AS BIGINT), NULL
            """
        else:
            target = f"{schema}.filtered_posts"
            select_cols = """
                CAST(t.id AS TEXT), CAST(t.subreddit AS TEXT), CAST(t.title AS TEXT),
                CAST(t.selftext AS TEXT), CAST(t.url AS TEXT),
                CAST(t.created_utc AS BIGINT), CAST(t.num_comments AS BIGINT),
                CAST(t.score AS BIGINT), CAST(t.author AS TEXT),
                CAST(t.subreddit_id AS TEXT), CAST(t.over_18 AS BOOLEAN),
                CAST(t.retrieved_on AS BIGINT), NULL
            """

        con.execute("BEGIN")
        con.execute(f"""
            INSERT INTO {target}
            SELECT {select_cols}
            FROM read_json_auto(?, format='newline_delimited', records='true') AS t
            INNER JOIN {schema}.valid_subs v ON lower(t.subreddit) = v.subreddit
        """, [str(f)])
        con.execute("COMMIT")

        upsert_state(con, schema, state_table, f, "done")
        print(f"  Ingested: {f.name}")

    con.close()
    print(f"Processed {len(files)} {args.mode} files")


if __name__ == "__main__":
    main()

