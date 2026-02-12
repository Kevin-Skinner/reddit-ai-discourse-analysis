"""
Stage 02: Seed Valid Subreddits
================================
We populate a lookup table of subreddit names that we want to include during
ingestion. This acts as a whitelist -- when we read millions of rows from the
Arctic Shift dumps, we join against this table to keep only the communities
we care about. This is far more efficient than filtering after the fact.
"""

from pathlib import Path
import yaml
import duckdb


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def seed_db(db_path, subs_path, schema="reddit"):
    """Insert subreddit names from a text file into the valid_subs lookup table."""
    con = duckdb.connect(str(db_path))
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
    con.execute(f"CREATE TABLE IF NOT EXISTS {schema}.valid_subs (subreddit TEXT);")
    con.execute(f"DELETE FROM {schema}.valid_subs;")

    # Read subreddit names (one per line), normalize to lowercase
    with open(subs_path, "r") as f:
        rows = [(line.strip().lower(),) for line in f if line.strip()]

    con.executemany(f"INSERT INTO {schema}.valid_subs (subreddit) VALUES (?)", rows)
    con.execute(f"CREATE INDEX IF NOT EXISTS valid_subs_idx ON {schema}.valid_subs(subreddit);")
    con.close()
    print(f"Seeded {len(rows)} subreddits into {db_path}")


def main():
    cfg = load_config()
    schema = cfg["duckdb"].get("schema", "reddit")

    # Find the valid subs file
    subs_file = None
    for path in cfg["data"]["valid_subs_file"]:
        if Path(path).exists():
            subs_file = path
            break

    if not subs_file:
        raise FileNotFoundError("Could not find valid_subs file from config.")

    # Seed both databases
    for db_key in ["comments_db", "posts_db"]:
        seed_db(cfg["duckdb"][db_key], subs_file, schema)


if __name__ == "__main__":
    main()

