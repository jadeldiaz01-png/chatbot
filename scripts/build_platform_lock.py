from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from packaging.utils import canonicalize_name, parse_wheel_filename


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    wheel_dir = Path(args.wheels)
    output = Path(args.output)
    records: dict[str, tuple[str, str]] = {}

    for wheel in sorted(wheel_dir.glob("*.whl")):
        name, version, _build, _tags = parse_wheel_filename(wheel.name)
        canonical_name = canonicalize_name(name)
        if canonical_name in records:
            raise RuntimeError(f"duplicate wheel for {canonical_name}")
        records[canonical_name] = (str(version), sha256(wheel))

    if not records:
        raise RuntimeError("no wheel files found")

    lines = [
        "#",
        "# Linux/amd64 CPython 3.12 runtime lock.",
        "# Generated from wheels downloaded for the pinned production dependency graph.",
        "# Install with: pip install --only-binary=:all: --require-hashes -r requirements.lock",
        "#",
    ]
    for name in sorted(records):
        version, digest = records[name]
        lines.append(f"{name}=={version} --hash=sha256:{digest}")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
