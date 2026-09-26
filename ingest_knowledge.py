#!/usr/bin/env python3
"""
AviGPT-250M NVMe Hardware Memory Bus Ingestion Tool
Architect & Creator: Yadlapalli Avinash Ricky

Allows users to instantly ingest new facts, documents, or entire folders into the
NVMe SSD database with zero retraining.
"""

import os
import sys
import argparse
import glob

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Universal cross-platform UTF-8 stream resilience (Windows, Linux CI/CD, macOS)
for _stream_name in ("stdout", "stderr", "stdin"):
    _stream = getattr(sys, _stream_name, None)
    if _stream and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from memory_bus import SSDMemoryEngine


def ingest_text(engine: SSDMemoryEngine, title: str, content: str, domain: str = "General") -> bool:
    clean_title = title.strip()
    clean_content = content.strip()
    if not clean_title or not clean_content:
        print("[!] Error: Both title and content must be non-empty.")
        return False

    success = engine.store(title=clean_title, content=clean_content, domain=domain)
    if success:
        print(f"[+] [SUCCESS] Ingested '{clean_title}' ({len(clean_content)} chars) into NVMe SSD memory in sub-millisecond time!")
    else:
        print(f"[-] [FAILED] Could not ingest '{clean_title}'.")
    return success


def ingest_file(engine: SSDMemoryEngine, file_path: str, title: str = None, domain: str = "Documents") -> bool:
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False

    file_title = title or os.path.splitext(os.path.basename(file_path))[0].replace("_", " ").title()
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return ingest_text(engine, file_title, content, domain)
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        return False


def ingest_folder(engine: SSDMemoryEngine, folder_path: str, domain: str = "Batch-Import"):
    if not os.path.isdir(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return

    exts = ["*.txt", "*.md", "*.json", "*.csv"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not files:
        print(f"⚠️ No compatible text files (.txt, .md, .json, .csv) found in {folder_path}")
        return

    print(f"🚀 Batch ingesting {len(files)} documents into NVMe SSD Memory Bus...")
    count = 0
    for f in files:
        if ingest_file(engine, f, domain=domain):
            count += 1
    print(f"\n🎉 Finished! Successfully added {count}/{len(files)} documents to AviGPT-250M memory.")


def main():
    parser = argparse.ArgumentParser(
        description="AviGPT-250M Flash Knowledge Ingestion CLI (Architect: Yadlapalli Avinash Ricky)"
    )
    parser.add_argument("--title", type=str, help="Title / entity name for the knowledge entry")
    parser.add_argument("--content", type=str, help="Text content / factual description")
    parser.add_argument("--file", type=str, help="Path to a text/markdown file to ingest")
    parser.add_argument("--folder", type=str, help="Path to a directory containing text files to batch ingest")
    parser.add_argument("--domain", type=str, default="User-Knowledge", help="Category / domain tag (default: User-Knowledge)")
    parser.add_argument("--db", type=str, default="avigpt_ssd_memory.db", help="Path to SQLite database file")

    args = parser.parse_args()

    # If no arguments provided, launch interactive prompt
    if not args.title and not args.file and not args.folder:
        print("=" * 75)
        print("🧠 AviGPT-250M Flash Knowledge Ingestion CLI")
        print("   Architect & Creator: Yadlapalli Avinash Ricky")
        print("=" * 75)
        print("Instantly add new knowledge to AviGPT's NVMe Memory Bus with 0 retraining!\n")

        title = input("Enter Title / Topic (e.g., 'Project Orion'): ").strip()
        if not title:
            print("Title cannot be empty. Exiting.")
            return

        print("\nEnter Content / Information (press Enter twice or Ctrl+Z/D to finish):")
        lines = []
        while True:
            try:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            except EOFError:
                break
        content = "\n".join(lines).strip()

        if not content:
            print("Content cannot be empty. Exiting.")
            return

        engine = SSDMemoryEngine(db_path=args.db)
        ingest_text(engine, title, content, domain="Interactive-User")
        engine.close()
        return

    engine = SSDMemoryEngine(db_path=args.db)

    if args.title and args.content:
        ingest_text(engine, args.title, args.content, domain=args.domain)
    elif args.file:
        ingest_file(engine, args.file, title=args.title, domain=args.domain)
    elif args.folder:
        ingest_folder(engine, args.folder, domain=args.domain)

    engine.close()


if __name__ == "__main__":
    main()
