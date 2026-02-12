"""
Stage 07: Generate Gemma Embeddings
=====================================
We generate 768-dimensional embeddings using Google's Gemma (embeddinggemma-300m).
Rather than encoding raw text, we read the pre-computed token sequences from
Stage 06, pad them into batches, and pass them through the model.

The embedding strategy is **mean pooling**: we take the average of all token-level
hidden states, weighted by the attention mask (so padding tokens don't contribute).
This produces a single dense vector per document that captures its semantic content.

We process in two levels of chunking:
- Outer loop: 10,000-document chunks read from DuckDB (memory management)
- Inner loop: 64-document batches through the GPU (throughput optimization)

A date filter restricts to the target time window (configured in config.yaml).
"""

import gc
import numpy as np
import torch
import pyarrow as pa
import duckdb
from transformers import AutoModel, AutoConfig
from tqdm import tqdm
import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pooling(model_output, attention_mask):
    """
    Mean pooling over token embeddings, weighted by attention mask.
    This ensures padding tokens (mask=0) don't contribute to the document vector.
    """
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def embed_batch(model, batch_tokens):
    """Pad a batch of variable-length token sequences, run through model, mean-pool."""
    max_len = max(len(t) for t in batch_tokens)
    padded = np.zeros((len(batch_tokens), max_len), dtype=int)
    masks = np.zeros((len(batch_tokens), max_len), dtype=int)

    for i, seq in enumerate(batch_tokens):
        padded[i, :len(seq)] = seq
        masks[i, :len(seq)] = 1

    input_ids = torch.tensor(padded).to(DEVICE)
    attention_mask = torch.tensor(masks).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = mean_pooling(outputs, attention_mask)

    return embeddings.float().cpu().numpy()


def process_source(input_db, output_db, table_name, model, cfg):
    """Generate embeddings for one source (posts or comments)."""
    batch_size = cfg["model"]["embedding_batch_size"]
    chunk_size = cfg["model"]["embedding_chunk_size"]
    start_date = cfg["data"]["start_date"]
    end_date = cfg["data"]["end_date"]

    con_in = duckdb.connect(input_db, read_only=True)
    con_out = duckdb.connect(output_db)
    con_out.execute(f"CREATE TABLE IF NOT EXISTS {table_name}_vectors (id VARCHAR, vector FLOAT[])")

    # Create a work queue filtered by date range
    con_in.execute(f"""
        CREATE TEMP TABLE work_queue AS
        SELECT id, tokens FROM reddit.{table_name}_tokenized
        WHERE created_utc >= {start_date} AND created_utc < {end_date}
    """)
    total = con_in.execute("SELECT COUNT(*) FROM work_queue").fetchone()[0]
    print(f"  {table_name}: {total:,} documents to embed")

    if total == 0:
        con_in.close()
        con_out.close()
        return

    pbar = tqdm(total=total, unit="docs")
    offset = 0

    while True:
        df = con_in.execute(f"SELECT * FROM work_queue LIMIT {chunk_size} OFFSET {offset}").df()
        if df.empty:
            break

        df = df.dropna(subset=['tokens'])
        df = df[df['tokens'].apply(lambda x: len(x) > 0)]
        if df.empty:
            offset += chunk_size
            pbar.update(chunk_size)
            continue

        ids = df['id'].tolist()
        tokens_list = df['tokens'].tolist()

        # Generate embeddings in GPU-sized batches
        vectors = []
        for i in range(0, len(df), batch_size):
            batch = tokens_list[i:i + batch_size]
            if batch:
                vectors.extend(embed_batch(model, batch))

        # Write to output database using PyArrow for efficiency
        arrow_table = pa.Table.from_pydict({'id': ids, 'vector': vectors})
        con_out.execute(f"INSERT INTO {table_name}_vectors SELECT * FROM arrow_table")

        offset += len(df)
        pbar.update(len(df))

        # Memory cleanup
        del df, arrow_table, vectors
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    pbar.close()
    con_in.close()
    con_out.close()


def main():
    cfg = load_config()
    model_id = cfg["model"]["embedding_model"]

    print(f"Loading model: {model_id} (device: {DEVICE})")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, config=config, trust_remote_code=True)
    model.to(DEVICE)
    model.eval()

    # Process posts
    print("Generating post embeddings...")
    process_source(
        cfg["duckdb"]["tokenized_posts_db"],
        cfg["duckdb"]["vectors_posts_db"],
        "posts", model, cfg
    )

    # Process comments
    print("Generating comment embeddings...")
    process_source(
        cfg["duckdb"]["tokenized_comments_db"],
        cfg["duckdb"]["vectors_comments_db"],
        "comments", model, cfg
    )


if __name__ == "__main__":
    main()

