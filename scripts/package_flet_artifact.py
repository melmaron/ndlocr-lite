#!/usr/bin/env python3
"""Create distributable .zip / .tar.gz archives and enforce size limits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tarfile
from typing import Optional, Sequence
import zipfile


DEFAULT_MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024  # GitHub Releases 2.00 GiB cap


def package_artifact(
    source_dir: Path,
    output_path: Path,
    *,
    max_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
    compresslevel: int = 6,
) -> int:
    """Package ``source_dir`` into ``output_path`` (.zip or .tar.gz) and return archive size."""
    root = Path(source_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {root}")

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    lower_name = out.name.lower()
    if lower_name.endswith(".zip"):
        # Write via Zip64 instead of PowerShell Compress-Archive (which fails near 2 GiB).
        with zipfile.ZipFile(
            out,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=compresslevel,
            allowZip64=True,
        ) as zf:
            top_name = root.name
            for item in sorted(root.rglob("*")):
                arcname = (Path(top_name) / item.relative_to(root)).as_posix()
                if item.is_dir():
                    zf.mkdir(arcname)
                elif item.is_file():
                    zf.write(item, arcname=arcname)
    elif lower_name.endswith(".tar.gz") or lower_name.endswith(".tgz"):
        with tarfile.open(out, "w:gz", compresslevel=compresslevel) as tf:
            tf.add(root, arcname=root.name)
    else:
        raise ValueError(
            f"Unsupported archive format for {out.name}; expected .zip, .tar.gz, or .tgz"
        )

    size_bytes = out.stat().st_size
    if max_bytes > 0 and size_bytes >= max_bytes:
        raise RuntimeError(
            f"Archive {out.name} size ({size_bytes:,} bytes, {size_bytes / (1024 ** 3):.2f} GiB) "
            f"meets or exceeds limit ({max_bytes:,} bytes)"
        )
    return size_bytes


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Package a built Flet output directory into .zip or .tar.gz with size check."
    )
    parser.add_argument("--source-dir", required=True, help="Directory to package (e.g. windows or linux)")
    parser.add_argument("--output", required=True, help="Output archive path (.zip or .tar.gz)")
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_ARCHIVE_BYTES,
        help="Strict upper bound on compressed archive size in bytes (default: 2 GiB)",
    )
    parser.add_argument(
        "--compresslevel",
        type=int,
        default=6,
        help="Compression level (default: 6)",
    )
    args = parser.parse_args(argv)

    try:
        size_bytes = package_artifact(
            Path(args.source_dir),
            Path(args.output),
            max_bytes=args.max_bytes,
            compresslevel=args.compresslevel,
        )
    except Exception as exc:
        print(f"[FAIL] Packaging failed: {exc}", file=sys.stderr)
        return 1

    mib = size_bytes / (1024 * 1024)
    gib = size_bytes / (1024 * 1024 * 1024)
    print(f"[OK] Packaged {args.output}: {size_bytes:,} bytes ({mib:.1f} MiB / {gib:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
