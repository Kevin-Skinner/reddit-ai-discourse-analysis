"""
Stage 01: Initialize DuckDB Databases
======================================
We create the DuckDB databases and schema that the rest of the pipeline writes into.
DuckDB is an in-process analytical database -- think SQLite for analytics. It handles
columnar storage, compressed reads, and SQL queries without a server.

We create separate databases for posts and comments, each with:
- A filtered content table (where ingested data lands)
- A state-tracking table (for resumable ingestion)
"""

from pathlib import Path
import yaml
import duckdb


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_schema_and_tables(con, schema):
    """Create the reddit schema with tables for posts, comments, and ingestion state."""
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")

    # Comments table
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}.filtered_comments (
            id              TEXT,
            subreddit       TEXT,
            body            TEXT,
            link_id         TEXT,
            parent_id       TEXT,
            created_utc     BIGINT,
            score           BIGINT,
            author          TEXT,
            subreddit_id    TEXT,
            retrieved_on    BIGINT,
            raw             JSON
        );
    """)

    # Posts table
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}.filtered_posts (
            id              TEXT,
            subreddit       TEXT,
            title           TEXT,
            selftext        TEXT,
            url             TEXT,
            created_utc     BIGINT,
            num_comments    BIGINT,
            score           BIGINT,
            author          TEXT,
            subreddit_id    TEXT,
            over_18         BOOLEAN,
            retrieved_on    BIGINT,
            raw             JSON
        );
    """)

    # State tables for tracking ingestion progress (enables resumability)
    for table_name in ["comments_state", "posts_state"]:
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.{table_name} (
                filepath    TEXT PRIMARY KEY,
                last_line   BIGINT,
                status      TEXT,
                updated_at  TIMESTAMP
            );
        """)


def main():
    cfg = load_config()
    schema = cfg["duckdb"].get("schema", "reddit")

    for db_key in ["comments_db", "posts_db"]:
        db_path = Path(cfg["duckdb"][db_key])
        db_path.parent.mkdir(parents=True, exist_ok=True)

        con = duckdb.connect(str(db_path))
        create_schema_and_tables(con, schema)
        con.close()
        print(f"Initialized: {db_path}")


if __name__ == "__main__":
    main()

