"""Download ONNX weights for HyLabel's AI Magic Wand (EdgeSAM and/or SAM2/SAM2.1)."""
import argparse
import os
import sys
import shutil
import urllib.request
import zipfile

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "labeler", "weights")

# Each entry: (repo_id, repo_type, hf_file, dest_filename)
MODEL_FILES = {
    "edgesam": [
        ("chongzhou/EdgeSAM", "space", "weights/edge_sam_3x_encoder.onnx", "edge_sam_3x_encoder.onnx"),
        ("chongzhou/EdgeSAM", "space", "weights/edge_sam_3x_decoder.onnx", "edge_sam_3x_decoder.onnx"),
    ],
    "sam2": [
        ("vietanhdev/segment-anything-2-onnx-models", "model", "sam2_hiera_tiny.encoder.onnx", "sam2_hiera_tiny.encoder.onnx"),
        ("vietanhdev/segment-anything-2-onnx-models", "model", "sam2_hiera_tiny.decoder.onnx", "sam2_hiera_tiny.decoder.onnx"),
    ],
    "sam2_base_plus": [
        ("vietanhdev/segment-anything-2-onnx-models", "model", "sam2_hiera_base_plus.encoder.onnx", "sam2_hiera_base_plus.encoder.onnx"),
        ("vietanhdev/segment-anything-2-onnx-models", "model", "sam2_hiera_base_plus.decoder.onnx", "sam2_hiera_base_plus.decoder.onnx"),
    ],
    "sam2_large": [
        ("vietanhdev/segment-anything-2-onnx-models", "model", "sam2_hiera_large.encoder.onnx", "sam2_hiera_large.encoder.onnx"),
        ("vietanhdev/segment-anything-2-onnx-models", "model", "sam2_hiera_large.decoder.onnx", "sam2_hiera_large.decoder.onnx"),
    ],
}

# SAM 2.1 is published only as zip archives, unlike the loose .onnx files above.
# Each entry: (repo_id, repo_type, hf_file, [member names to extract])
MODEL_ARCHIVES = {
    "sam2_1_large": [
        (
            "vietanhdev/segment-anything-2.1-onnx-models", "model",
            "sam2.1_hiera_large_20260221.zip",
            ["sam2.1_hiera_large.encoder.onnx", "sam2.1_hiera_large.decoder.onnx"],
        ),
    ],
}

ALL_MODELS = list(MODEL_FILES.keys()) + list(MODEL_ARCHIVES.keys())

HF_MANUAL_URLS = {
    "edgesam": "https://huggingface.co/spaces/chongzhou/EdgeSAM",
    "sam2": "https://huggingface.co/vietanhdev/segment-anything-2-onnx-models",
    "sam2_base_plus": "https://huggingface.co/vietanhdev/segment-anything-2-onnx-models",
    "sam2_large": "https://huggingface.co/vietanhdev/segment-anything-2-onnx-models",
    "sam2_1_large": "https://huggingface.co/vietanhdev/segment-anything-2.1-onnx-models",
}


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    done = block_num * block_size
    if total_size > 0:
        pct = min(done / total_size * 100, 100)
        bar = "#" * int(pct // 2)
        print(f"\r[{bar:<50}] {pct:5.1f}%  ({done/1e6:.1f}/{total_size/1e6:.1f} MB)",
              end="", flush=True)
    else:
        print(f"\r  Downloaded {done/1e6:.1f} MB", end="", flush=True)


def _download_one(repo_id: str, repo_type: str, hf_file: str, dest: str) -> bool:
    """Try huggingface_hub first, fall back to urllib. Returns True on success."""
    if os.path.exists(dest):
        print(f"  Already exists: {os.path.basename(dest)}")
        return True

    # Try huggingface_hub
    try:
        from huggingface_hub import hf_hub_download
        print(f"  Downloading via huggingface_hub: {os.path.basename(dest)} …")
        cached = hf_hub_download(repo_id=repo_id, filename=hf_file, repo_type=repo_type)
        shutil.copy(cached, dest)
        print(f"  Saved: {dest}")
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"  huggingface_hub failed: {e}")

    # Fallback: direct URL
    prefix = "spaces/" if repo_type == "space" else ""
    url = f"https://huggingface.co/{prefix}{repo_id}/resolve/main/{hf_file}"
    print(f"  Downloading via urllib:\n    {url}\n    → {dest}\n")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print(f"\n  Saved: {dest}")
        return True
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        print(f"\n  Error: {e}")
        return False


def _download_archive(repo_id: str, repo_type: str, hf_file: str,
                      members: list) -> bool:
    """Fetch a zip and extract the named members into WEIGHTS_DIR."""
    if all(os.path.exists(os.path.join(WEIGHTS_DIR, m)) for m in members):
        print(f"  Already exists: {', '.join(members)}")
        return True

    tmp_zip = os.path.join(WEIGHTS_DIR, os.path.basename(hf_file))
    if not _download_one(repo_id, repo_type, hf_file, tmp_zip):
        return False

    try:
        with zipfile.ZipFile(tmp_zip) as z:
            names = z.namelist()
            for m in members:
                if m not in names:
                    print(f"  Error: {m} not found in {os.path.basename(hf_file)}")
                    return False
                z.extract(m, WEIGHTS_DIR)
                print(f"  Extracted: {m}")
    except Exception as e:
        print(f"  Error extracting {os.path.basename(hf_file)}: {e}")
        return False
    finally:
        # The archive is several hundred MB — don't leave a second copy behind.
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)
    return True


def download_model(model_key: str) -> list:
    """Download one model's weight files. Returns list of failed filenames."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    failed = []
    print(f"\n=== {model_key} ===")
    for repo_id, repo_type, hf_file, filename in MODEL_FILES.get(model_key, []):
        dest = os.path.join(WEIGHTS_DIR, filename)
        if not _download_one(repo_id, repo_type, hf_file, dest):
            failed.append(filename)
    for repo_id, repo_type, hf_file, members in MODEL_ARCHIVES.get(model_key, []):
        if not _download_archive(repo_id, repo_type, hf_file, members):
            failed.extend(members)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=ALL_MODELS + ["all"], default="edgesam",
        help="Which model's weights to download (default: edgesam)",
    )
    args = parser.parse_args()

    models = ALL_MODELS if args.model == "all" else [args.model]

    all_failed = {}
    for model_key in models:
        failed = download_model(model_key)
        if failed:
            all_failed[model_key] = failed

    if all_failed:
        print("\nFailed to download:")
        for model_key, files in all_failed.items():
            print(f"  {model_key}:")
            for f in files:
                print(f"    {f}")
            print(
                f"\n  Manual download:"
                f"\n    1) {HF_MANUAL_URLS[model_key]} → Files"
                f"\n    2) Place the .onnx files in: {WEIGHTS_DIR}"
            )
        sys.exit(1)
    else:
        print("\nAll weights ready.")


if __name__ == "__main__":
    main()
