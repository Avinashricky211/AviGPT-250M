"""
=============================================================================
AviGPT-250M: Universal Dataset Ingestion Cookbook
=============================================================================
Architect & Creator: Yadlapalli Avinash Ricky
Project: AviGPT-250M High-Efficiency Edge SLM

This cookbook provides developers and users with a universal template to convert
and ingest ANY Hugging Face dataset, JSONL, CSV, or Parquet file directly into
AviGPT's native NVMe SSD Hardware Memory Bus (`avigpt_ssd_memory.db`).

Zero retraining. Instant sub-millisecond retrieval.
=============================================================================
"""

import os
import sys
import json
import sqlite3
import argparse
from typing import Optional, Dict, Any, Generator

# Ensure resilient console encoding across all Windows and Linux environments
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def get_db_connection(db_path: str = "avigpt_ssd_memory.db") -> sqlite3.Connection:
    """Connect to AviGPT NVMe database and ensure FTS5 table exists."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS ssd_knowledge USING fts5(
            title,
            content,
            domain,
            tokenize = 'porter unicode61'
        );
    """)
    conn.commit()
    return conn


def batch_insert(conn: sqlite3.Connection, records: list) -> int:
    """Fast batch insert into FTS5 virtual table."""
    if not records:
        return 0
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO ssd_knowledge(title, content, domain) VALUES (?, ?, ?);",
        records
    )
    conn.commit()
    return len(records)


# =============================================================================
# TEMPLATE 1: Wikipedia / Encyclopedic Datasets (e.g., wikimedia/wikipedia)
# =============================================================================
def ingest_wikipedia(conn: sqlite3.Connection, limit: int = 5000):
    """
    Template for Encyclopedic Knowledge.
    Maps:
      item['title'] -> title
      item['text'] -> content (Full length preserved with zero semantic loss)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[!] Please install datasets: pip install datasets")
        return

    print(f"[*] Streaming Wikipedia (wikimedia/wikipedia) - Target: {limit} articles...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    batch = []
    total = 0
    for i, item in enumerate(dataset):
        title = item.get("title", "").strip()
        text = item.get("text", "").strip()
        if not title or not text:
            continue

        clean_text = text.replace("\n", " ").strip()
        batch.append((title, clean_text, "Encyclopedia"))

        if len(batch) >= 500:
            total += batch_insert(conn, batch)
            print(f"    Indexed {total:,} Wikipedia articles into NVMe bus...")
            batch = []

        if total >= limit:
            break

    if batch:
        total += batch_insert(conn, batch)
    print(f"[OK] Successfully ingested {total:,} Wikipedia articles into NVMe memory.")


# =============================================================================
# TEMPLATE 2: Scientific Research / Papers (e.g., secemp9/arxiv-complete)
# =============================================================================
def ingest_arxiv(conn: sqlite3.Connection, limit: int = 5000):
    """
    Template for Academic Papers and Abstracts.
    Maps:
      item['title'] -> title
      item['abstract'] -> content
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[!] Please install datasets: pip install datasets")
        return

    print(f"[*] Streaming arXiv Papers (secemp9/arxiv-complete) - Target: {limit} papers...")
    dataset = load_dataset("secemp9/arxiv-complete", split="train", streaming=True)

    batch = []
    total = 0
    for i, item in enumerate(dataset):
        title = item.get("title", "").strip().replace("\n", " ")
        abstract = item.get("abstract", "").strip().replace("\n", " ")
        categories = item.get("categories", "Research")

        if not title or not abstract:
            continue

        content = f"Abstract: {abstract}"
        batch.append((title, content, f"arXiv:{categories[:15]}"))

        if len(batch) >= 500:
            total += batch_insert(conn, batch)
            print(f"    Indexed {total:,} scientific papers into NVMe bus...")
            batch = []

        if total >= limit:
            break

    if batch:
        total += batch_insert(conn, batch)
    print(f"[OK] Successfully ingested {total:,} scientific papers into NVMe memory.")


# =============================================================================
# TEMPLATE 3: Reasoning / Solution Cookbook (e.g., Fable-5.1-Reasoning)
# =============================================================================
def ingest_reasoning_cookbook(
    conn: sqlite3.Connection,
    hf_dataset_id: str = "MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x",
    limit: int = 5000
):
    """
    Template for Reasoning / Chain-of-Thought Cookbooks.
    Stores worked-out reasoning solutions so AviGPT can retrieve them during <think>
    as immediate few-shot guides without neural retraining.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[!] Please install datasets: pip install datasets")
        return

    print(f"[*] Streaming Reasoning Cookbook ({hf_dataset_id}) - Target: {limit} examples...")
    dataset = load_dataset(hf_dataset_id, split="train", streaming=True)

    batch = []
    total = 0
    for i, item in enumerate(dataset):
        prompt = (
            item.get("prompt") or 
            item.get("question") or 
            item.get("instruction") or 
            item.get("problem") or ""
        ).strip()
        
        reasoning = (
            item.get("reasoning") or 
            item.get("thought") or 
            item.get("solution") or 
            item.get("response") or 
            item.get("output") or ""
        ).strip()

        if not prompt or not reasoning:
            continue

        # Format as a cognitive solution card
        title = prompt[:120].replace("\n", " ").strip()
        content = f"Problem: {prompt}\n\nWorked Solution:\n{reasoning}"
        batch.append((title, content, "ReasoningCookbook"))

        if len(batch) >= 500:
            total += batch_insert(conn, batch)
            print(f"    Indexed {total:,} reasoning recipes into NVMe bus...")
            batch = []

        if total >= limit:
            break

    if batch:
        total += batch_insert(conn, batch)
    print(f"[OK] Successfully ingested {total:,} reasoning solutions into NVMe memory.")


# =============================================================================
# TEMPLATE 4: Universal File Loader (JSONL, CSV, JSON)
# =============================================================================
def ingest_custom_file(
    conn: sqlite3.Connection,
    file_path: str,
    title_key: str,
    content_key: str,
    domain: str = "Custom"
):
    """
    Universal ingestion for local data files (JSONL, JSON, CSV).
    Allows mapping any column to Title and Content.
    """
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return

    print(f"[*] Loading custom dataset from '{file_path}' (Domain: {domain})...")
    batch = []
    total = 0

    if file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            skipped = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    title = str(data.get(title_key, "")).strip()
                    content = str(data.get(content_key, "")).strip()
                    if title and content:
                        batch.append((title, content, domain))
                except Exception:
                    skipped += 1
                    continue

                if len(batch) >= 500:
                    total += batch_insert(conn, batch)
                    batch = []
        if skipped:
            print(f"[!] Warning: {skipped} line(s) skipped during JSONL loading (malformed JSON).")

    elif file_path.endswith(".csv"):
        import csv
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = str(row.get(title_key, "")).strip()
                content = str(row.get(content_key, "")).strip()
                if title and content:
                    batch.append((title, content, domain))

                if len(batch) >= 500:
                    total += batch_insert(conn, batch)
                    batch = []

    elif file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)
            if isinstance(items, list):
                for item in items:
                    title = str(item.get(title_key, "")).strip()
                    content = str(item.get(content_key, "")).strip()
                    if title and content:
                        batch.append((title, content, domain))
            total += batch_insert(conn, batch)
            batch = []

    if batch:
        total += batch_insert(conn, batch)

    print(f"[OK] Ingested {total:,} records from '{file_path}' into AviGPT's NVMe database.")


# =============================================================================
# CLI Interface
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="AviGPT-250M: Universal Dataset Ingestion Cookbook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest 2,000 Wikipedia articles:
  python dataset_cookbook.py --preset wikipedia --limit 2000

  # Ingest 2,000 ArXiv research papers:
  python dataset_cookbook.py --preset arxiv --limit 2000

  # Ingest 2,000 Fable-5.1 Reasoning Cookbook recipes:
  python dataset_cookbook.py --preset reasoning --limit 2000

  # Ingest a custom JSONL file:
  python dataset_cookbook.py --file my_docs.jsonl --title-col question --content-col answer --domain FAQ
        """
    )
    parser.add_argument("--db", type=str, default="avigpt_ssd_memory.db", help="Path to SQLite database")
    parser.add_argument("--preset", type=str, choices=["wikipedia", "arxiv", "reasoning"], help="Preset dataset pipeline")
    parser.add_argument("--hf-dataset", type=str, default=None, help="Custom HuggingFace dataset ID (for reasoning preset)")
    parser.add_argument("--limit", type=int, default=1000, help="Max records to stream and ingest")
    parser.add_argument("--file", type=str, help="Path to custom .jsonl, .json, or .csv file")
    parser.add_argument("--title-col", type=str, default="title", help="Column/key for Title")
    parser.add_argument("--content-col", type=str, default="content", help="Column/key for Content")
    parser.add_argument("--domain", type=str, default="General", help="Domain classification tag")

    args = parser.parse_args()

    conn = get_db_connection(args.db)

    if args.preset == "wikipedia":
        ingest_wikipedia(conn, limit=args.limit)
    elif args.preset == "arxiv":
        ingest_arxiv(conn, limit=args.limit)
    elif args.preset == "reasoning":
        hf_id = args.hf_dataset or "MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x"
        ingest_reasoning_cookbook(conn, hf_dataset_id=hf_id, limit=args.limit)
    elif args.file:
        ingest_custom_file(conn, args.file, args.title_col, args.content_col, domain=args.domain)
    else:
        parser.print_help()

    conn.close()


if __name__ == "__main__":
    main()
