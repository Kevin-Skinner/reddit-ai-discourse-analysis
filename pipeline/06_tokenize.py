"""
Stage 06: Tokenization
=======================
We tokenize filtered posts and comments using the Gemma tokenizer, storing
the resulting token ID sequences back into DuckDB. This pre-tokenization step
means we only run the tokenizer once -- downstream embedding generation can
read token IDs directly without re-tokenizing.

For posts, we concatenate title + selftext as the input text.
For comments, we use the body field.

The tokenizer is configured with:
- add_special_tokens=True (includes BOS/EOS markers)
- truncation at 512 tokens (Gemma's effective context window for embeddings)
- No padding (variable-length sequences, padded at embedding time)
"""

import os
import duckdb
import pandas as pd
from transformers import AutoTokenizer
import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


BATCH_SIZE = 10_000


def tokenize_table(db_path, source_table, dest_table, text_builder, model_id, max_length):
    """
    Tokenize all rows from source_table, writing results to dest_table.

    Args:
        db_path: Path to DuckDB database
        source_table: Source table name (e.g., 'reddit.ai_keyword_posts')
        dest_table: Destination table name (e.g., 'reddit.posts_tokenized')
        text_builder: SQL expression to build text from columns (e.g., "title || ' ' || selftext")
        model_id: HuggingFace model ID for tokenizer
        max_length: Maximum token sequence length
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    con = duckdb.connect(db_path)

    # Create destination table if it doesn't exist
    exists = con.execute(
        f"SELECT count(*) FROM information_schema.tables "
        f"WHERE table_name = '{dest_table.split('.')[-1]}'"
    ).fetchone()[0]

    if exists == 0:
        con.execute(f"CREATE TABLE {dest_table} AS SELECT * FROM {source_table} LIMIT 0")
        con.execute(f"ALTER TABLE {dest_table} ADD COLUMN tokens INTEGER[]")

    total = con.execute(f"SELECT COUNT(*) FROM {source_table}").fetchone()[0]
    processed = con.execute(f"SELECT COUNT(*) FROM {dest_table}").fetchone()[0]
    offset = processed

    while offset < total:
        batch_df = con.execute(
            f"SELECT * FROM {source_table} LIMIT {BATCH_SIZE} OFFSET {offset}"
        ).df()
        if batch_df.empty:
            break

        # Build text column from the appropriate fields
        texts = (batch_df['title'].fillna("") + " " + batch_df['selftext'].fillna("")).astype(str).tolist() \
            if 'title' in batch_df.columns \
            else batch_df['body'].fillna("").astype(str).tolist()

        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False
        )
        batch_df['tokens'] = encoded['input_ids']
        con.execute(f"INSERT INTO {dest_table} SELECT * FROM batch_df")

        offset += len(batch_df)
        print(f"  Tokenized {offset:,}/{total:,} rows")

    con.close()


def main():
    cfg = load_config()
    model_id = cfg["model"]["embedding_model"]
    max_length = cfg["model"]["max_token_length"]

    # Tokenize posts
    print("Tokenizing posts...")
    tokenize_table(
        cfg["duckdb"]["filtered_posts_db"],
        "reddit.ai_keyword_posts",
        "reddit.posts_tokenized",
        "title || ' ' || selftext",
        model_id, max_length
    )

    # Tokenize comments
    print("Tokenizing comments...")
    tokenize_table(
        cfg["duckdb"]["filtered_comments_db"],
        "reddit.ai_keyword_comments",
        "reddit.comments_tokenized",
        "body",
        model_id, max_length
    )


if __name__ == "__main__":
    main()

