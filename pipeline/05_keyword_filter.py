"""
Stage 05: Keyword Filtering Pipeline
======================================
We filter posts and comments to those containing AI-related terms using a
two-stage approach optimized for large datasets:

Stage 1 (LIKE prefilter):
    A fast SQL LIKE scan over text columns using the most common terms.
    This is cheap but imprecise -- it narrows millions of rows to a smaller
    candidate set quickly.

Stage 2 (Regex refinement):
    Word-boundary regex matching against the full term list. This is precise
    but expensive, so running it only on the candidate set from Stage 1
    keeps total processing time manageable.

The keyword terms are loaded from a user-supplied file (configured in config.yaml).
We intentionally do not include the specific terms in this repository -- users
should supply their own domain-specific keyword list.
"""

import pickle
import re
import math
import duckdb
import pandas as pd
import yaml


BATCH_SIZE = 2_000_000
MAX_LIKE_TERMS = 20  # Number of terms to use for the fast LIKE prefilter


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_terms(terms_path):
    """Load keyword terms from a pickle file. Normalize and deduplicate."""
    with open(terms_path, "rb") as f:
        terms = pickle.load(f)

    if isinstance(terms, dict):
        terms = list(terms.keys())
    if isinstance(terms, (set, tuple)):
        terms = list(terms)

    # Normalize: lowercase, strip, remove very short terms (except 'ai')
    keep = []
    for t in terms:
        s = str(t).strip().lower()
        if not s or (len(s) < 3 and s != "ai"):
            continue
        keep.append(s)

    keep = sorted(set(keep))
    keep_regex = [re.escape(s) for s in keep]
    return keep, keep_regex


def setup_output_db(path, terms_regex, tmp_dir):
    """Create the output database and store the term list for regex matching."""
    con = duckdb.connect(path)
    con.execute("CREATE SCHEMA IF NOT EXISTS reddit")
    con.execute(f"PRAGMA temp_directory='{tmp_dir}'")
    con.execute("PRAGMA memory_limit='12GB'")

    # Store terms in a table so we can join against them in SQL
    con.execute("DROP TABLE IF EXISTS reddit.ai_terms")
    con.register("terms_df", pd.DataFrame({"term": terms_regex}))
    con.execute("CREATE TABLE reddit.ai_terms AS SELECT term FROM terms_df")
    con.unregister("terms_df")
    con.close()


def filter_table(src_db, out_db, kind, like_terms, terms_regex, tmp_dir):
    """
    Two-stage filtering:
    1. LIKE prefilter on text columns to get candidates
    2. Regex word-boundary match against full term list
    """
    con = duckdb.connect()
    con.execute(f"ATTACH DATABASE '{src_db}' AS src (READ_ONLY)")
    con.execute(f"ATTACH DATABASE '{out_db}' AS dest")
    con.execute(f"PRAGMA temp_directory='{tmp_dir}'")

    table = f"src.reddit.filtered_{'posts' if kind == 'posts' else 'comments'}"
    alias = "p" if kind == "posts" else "c"
    total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    batches = math.ceil(total / BATCH_SIZE)

    # Build LIKE clauses for the prefilter
    like_clauses = []
    for t in like_terms[:MAX_LIKE_TERMS]:
        like_clauses.append(f"lower({alias}.subreddit) LIKE '%{t}%'")
        if kind == "posts":
            like_clauses.append(f"lower({alias}.title) LIKE '%{t}%'")
            like_clauses.append(f"lower({alias}.selftext) LIKE '%{t}%'")
        else:
            like_clauses.append(f"lower({alias}.body) LIKE '%{t}%'")
    like_prefilter = " OR ".join(like_clauses)

    # Create destination table
    dest_table = f"dest.reddit.ai_keyword_{kind}"
    con.execute(f"DROP TABLE IF EXISTS {dest_table}")
    con.execute(f"CREATE TABLE {dest_table} AS SELECT * FROM {table} WHERE 1=0")

    for i in range(batches):
        offset = i * BATCH_SIZE

        # Stage 1: LIKE prefilter
        subq = f"(SELECT * FROM {table} LIMIT {BATCH_SIZE} OFFSET {offset}) AS {alias}"
        con.execute(f"CREATE TEMP TABLE tmp_cand AS SELECT * FROM {subq} WHERE {like_prefilter}")
        cand = con.execute("SELECT COUNT(*) FROM tmp_cand").fetchone()[0]

        if cand > 0:
            # Stage 2: Regex refinement with word boundaries
            text_col = f"lower({alias}.title)" if kind == "posts" else f"lower({alias}.body)"
            regex_where = (
                f"EXISTS (SELECT 1 FROM dest.reddit.ai_terms t "
                f"WHERE regexp_matches({text_col}, ('\\\\b' || t.term || '\\\\b')))"
            )
            con.execute(f"CREATE TEMP TABLE tmp_filt AS SELECT * FROM tmp_cand {alias} WHERE {regex_where}")
            hits = con.execute("SELECT COUNT(*) FROM tmp_filt").fetchone()[0]
            if hits > 0:
                con.execute(f"INSERT INTO {dest_table} SELECT * FROM tmp_filt")
            con.execute("DROP TABLE IF EXISTS tmp_filt")

        con.execute("DROP TABLE IF EXISTS tmp_cand")
        print(f"  Batch {i+1}/{batches}: {cand} candidates")

    final = con.execute(f"SELECT COUNT(*) FROM {dest_table}").fetchone()[0]
    con.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{kind}_id ON {dest_table}(id)")
    print(f"Finished {kind}: {final:,} rows")
    con.close()


def main():
    cfg = load_config()
    terms_path = cfg["data"]["keyword_terms_file"]
    tmp_dir = cfg["data"].get("tmp_dir", "/tmp/duckdb_tmp")
    like_terms, terms_regex = load_terms(terms_path)

    out_posts = cfg["duckdb"]["filtered_posts_db"]
    out_comments = cfg["duckdb"]["filtered_comments_db"]

    setup_output_db(out_posts, terms_regex, tmp_dir)
    setup_output_db(out_comments, terms_regex, tmp_dir)

    filter_table(cfg["duckdb"]["posts_db"], out_posts, "posts", like_terms, terms_regex, tmp_dir)
    filter_table(cfg["duckdb"]["comments_db"], out_comments, "comments", like_terms, terms_regex, tmp_dir)


if __name__ == "__main__":
    main()

