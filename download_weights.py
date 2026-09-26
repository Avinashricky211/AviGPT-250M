"""
AviGPT-250M: 1-Click Weights & NVMe Memory Bus Downloader
--------------------------------------------------------
Downloads official model weights (477 MB) and pre-indexed NVMe memory bus (55 MB)
directly from the official Hugging Face Hub repository.

Architect & Creator: Yadlapalli Avinash Ricky (India)
"""

import os
import sys

REPO_ID = "AvinashRicky/avigpt-250m-instruct"


def download_assets():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[*] Installing required 'huggingface_hub' package...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download

    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 75)
    print("⚡ AVIGPT-250M-INSTRUCT: 1-CLICK WEIGHTS & MEMORY BUS SETUP")
    print(f"   Source: https://huggingface.co/{REPO_ID}")
    print("=" * 75)

    # 1. Download Checkpoint Weights
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "avigpt_250m_instruct.pt")
    if not os.path.exists(ckpt_path):
        print("\n[1/2] Downloading 'avigpt_250m_instruct.pt' (~477 MB)...")
        os.makedirs(ckpt_dir, exist_ok=True)
        hf_hub_download(
            repo_id=REPO_ID,
            filename="checkpoints/avigpt_250m_instruct.pt",
            local_dir=base_dir,
        )
        print("      ✅ Checkpoint weights downloaded successfully!")
    else:
        print("\n[1/2] ✅ Checkpoint weights already present.")

    # 2. Download SQLite NVMe DB
    db_path = os.path.join(base_dir, "avigpt_ssd_memory.db")
    if not os.path.exists(db_path):
        print("\n[2/2] Downloading 'avigpt_ssd_memory.db' (~55 MB)...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename="avigpt_ssd_memory.db",
            local_dir=base_dir,
        )
        print("      ✅ NVMe Memory Bus database downloaded successfully!")
    else:
        print("\n[2/2] ✅ NVMe Memory Bus database already present.")

    print("\n" + "=" * 75)
    print("🎉 ALL ASSETS READY! You can now run AviGPT-250M:")
    print("   • Terminal (Interactive): python terminal_eval.py --interactive")
    print("   • Benchmark Suite:        python eval_proof.py")
    print("   • Windows 1-Click:        launch_terminal.bat")
    print("=" * 75)


if __name__ == "__main__":
    download_assets()
