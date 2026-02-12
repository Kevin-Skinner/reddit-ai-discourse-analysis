"""
Stage 19: Extract Dominant Topics per Subreddit
=================================================
We extract the single most dominant topic for each subreddit (by document count)
along with its characterizing keywords. This provides a quick-reference summary
of what each community primarily discusses about AI.

The output CSV can be used for:
- A table in the README or presentation
- Labeling nodes in network visualizations
- Quick comparison of community focus areas
"""

import os
import pandas as pd
import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    export_dir = cfg["output"]["data_exports_dir"]

    # Load topic details (produced during topic modeling -- all_topic_details.csv)
    details_path = os.path.join(export_dir, "all_topic_details.csv")
    if not os.path.exists(details_path):
        print(f"Topic details not found at {details_path}")
        return

    df = pd.read_csv(details_path)
    print(f"Loaded {len(df)} topic records")

    # Filter to subreddit models only
    df_subs = df[df["model_type"] == "subreddit"].copy()
    df_subs["subreddit"] = df_subs["model_name"].str.replace("subreddit_", "", n=1)
    print(f"  {len(df_subs)} subreddit topic records")

    # Totals per subreddit for percentage calculation
    sub_totals = df_subs.groupby("subreddit")["document_count"].sum().to_dict()

    # Dominant topic = highest document_count per subreddit
    idx = df_subs.groupby("subreddit")["document_count"].idxmax()
    dominant = df_subs.loc[idx].copy()

    dominant["total_docs"] = dominant["subreddit"].map(sub_totals)
    dominant["pct_of_subreddit"] = (
        dominant["document_count"] / dominant["total_docs"] * 100
    ).round(1)

    result = dominant[[
        "subreddit", "topic_id", "document_count", "total_docs",
        "pct_of_subreddit", "top_words"
    ]].rename(columns={
        "topic_id": "dominant_topic_id",
        "document_count": "topic_doc_count",
        "total_docs": "subreddit_total_docs",
    }).sort_values("subreddit").reset_index(drop=True)

    out_path = os.path.join(export_dir, "subreddit_dominant_topics.csv")
    result.to_csv(out_path, index=False)
    print(f"Saved {len(result)} subreddits to {out_path}")

    # Preview
    print("\nSample:")
    for _, r in result.head(5).iterrows():
        words = str(r.get("top_words", ""))[:60]
        print(f"  r/{r['subreddit']}: Topic #{r['dominant_topic_id']} "
              f"({r['topic_doc_count']:,} docs, {r['pct_of_subreddit']}%) -- {words}")


if __name__ == "__main__":
    main()

