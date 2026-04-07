"""Download the CAM++ zh-cn common checkpoint from ModelScope.

Usage:
    uv run python download_campplus.py [--out-dir assets/campplus]

The ModelScope file-download endpoint works with a plain HTTP GET for public
models, so we don't need the `modelscope` SDK at runtime — just `requests`.
The downloaded file is ``campplus_cn_common.bin`` (≈28 MB) and is the exact
checkpoint used by ``iic/speech_campplus_sv_zh-cn_16k-common`` (revision
v1.0.0, embedding_size=192, feat_dim=80).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests


MODEL_ID = "iic/speech_campplus_sv_zh-cn_16k-common"
REVISION = "v1.0.0"
FILE_NAME = "campplus_cn_common.bin"

# ModelScope public file-download endpoint (no auth needed for public models).
URL_TEMPLATE = (
    "https://modelscope.cn/api/v1/models/{model_id}/repo"
    "?Revision={revision}&FilePath={file_name}"
)


def download(out_dir: Path, chunk_size: int = 1 << 20) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / FILE_NAME
    if target.exists() and target.stat().st_size > 0:
        print(f"[skip] already exists: {target} ({target.stat().st_size / 1e6:.1f} MB)")
        return target

    url = URL_TEMPLATE.format(model_id=MODEL_ID, revision=REVISION, file_name=FILE_NAME)
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        written = 0
        tmp = target.with_suffix(".bin.part")
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    sys.stdout.write(
                        f"\r[download] {written / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.1f}%)"
                    )
                    sys.stdout.flush()
        sys.stdout.write("\n")
        tmp.rename(target)

    print(f"[done] {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "assets" / "campplus",
        help="Directory to save the checkpoint (default: ../assets/campplus)",
    )
    args = parser.parse_args()
    download(args.out_dir)


if __name__ == "__main__":
    main()
