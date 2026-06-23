"""Download EdgeSAM weights into labeler/weights/."""
import os
import sys
import urllib.request

DEST = os.path.join(os.path.dirname(__file__), "labeler", "weights",
                    "edge_sam_3x.pth")

HF_REPO  = "chongzhou/EdgeSAM"
HF_FILE  = "weights/edge_sam_3x.pth"
URL_DIRECT = (
    "https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/"
    "weights/edge_sam_3x.pth"
)


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    done = block_num * block_size
    if total_size > 0:
        pct = min(done / total_size * 100, 100)
        bar = "#" * int(pct // 2)
        print(f"\r[{bar:<50}] {pct:5.1f}%  ({done/1e6:.1f}/{total_size/1e6:.1f} MB)",
              end="", flush=True)
    else:
        print(f"\r  Downloaded {done/1e6:.1f} MB", end="", flush=True)


def _try_huggingface_hub() -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    print("Downloading via huggingface_hub…")
    try:
        cached = hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_FILE,
            repo_type="space",
        )
        import shutil
        shutil.copy(cached, DEST)
        return True
    except Exception as e:
        print(f"  huggingface_hub failed: {e}")
        return False


def main() -> None:
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    if os.path.exists(DEST):
        print(f"Already downloaded: {DEST}")
        return

    if _try_huggingface_hub():
        print(f"\nSaved to: {DEST}")
        return

    print(f"Downloading via urllib…\n  {URL_DIRECT}\n  → {DEST}\n")
    try:
        urllib.request.urlretrieve(URL_DIRECT, DEST, reporthook=_progress)
        print(f"\nDone! Saved to: {DEST}")
    except Exception as e:
        if os.path.exists(DEST):
            os.remove(DEST)
        print(f"\nError: {e}")
        print(
            f"\nManual download:"
            f"\n  1) https://huggingface.co/spaces/chongzhou/EdgeSAM"
            f"\n     → Files → weights/edge_sam_3x.pth"
            f"\n  2) Place it at: {DEST}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
