"""
Stage 04: Add Unique Indexes
==============================
We create unique indexes on the `id` columns of our filtered tables.
This serves two purposes:
1. Deduplication -- prevents duplicate records if we re-run ingestion
2. Performance -- dramatically speeds up the joins we do in later stages
"""

from pathlib import Path
import yaml
import duckdb


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def add_indexes(db_path, schema="reddit"):
    con = duckdb.connect(str(db_path))
    con.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS comments_id_uq ON {schema}.filtered_comments(id);")
    con.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS posts_id_uq    ON {schema}.filtered_posts(id);")
    con.close()
    print(f"Indexes added to {db_path}")


def main():
    cfg = load_config()
    schema = cfg["duckdb"].get("schema", "reddit")

    for db_key in ["comments_db", "posts_db"]:
        add_indexes(cfg["duckdb"][db_key], schema)


if __name__ == "__main__":
    main()

