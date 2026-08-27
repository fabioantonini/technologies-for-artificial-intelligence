"""Build, tag and publish the two course Docker images.

    python tools/release.py patch      # 0.1.0 -> 0.1.1
    python tools/release.py minor      # 0.1.0 -> 0.2.0
    python tools/release.py major      # 0.1.0 -> 1.0.0
    python tools/release.py patch --dry-run
    python tools/release.py patch --no-push

VERSION is the single source of truth. This script also rewrites the version
string wherever it appears in the documentation, so the student quickstart can
never drift out of step with what was actually published - which is exactly
the failure mode the previous course repository suffered from.

Two images, not one. `:core` carries lessons 1 to 8; `:full` is built FROM it
and adds TensorFlow for lessons 9 and 10, which is 1.3 GB nobody needs until
20 November. Because full is built on core rather than beside it, a student who
already has core downloads only the TensorFlow layer.

There is deliberately no `:latest`. With two images that name has no honest
meaning, and the one thing worse than a large download is the wrong image.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "VERSION"

# One image name, used here, in docker-compose.yml and in every document.
IMAGE = "fabioantonini/technologies-for-artificial-intelligence"

# Files whose version references are kept in sync with VERSION.
VERSIONED_DOCS = (
    "README.md",
    "Course/Setup/Docker_Quickstart.md",
    "docker-compose.yml",
)

SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def read_version() -> str:
    if not VERSION_FILE.exists():
        sys.exit(f"missing {VERSION_FILE.name} - it is the source of truth, create it")
    raw = VERSION_FILE.read_text(encoding="utf8").strip()
    if not SEMVER.match(raw):
        sys.exit(f"invalid VERSION {raw!r}, expected X.Y.Z")
    return raw


def bump(version: str, part: str) -> str:
    major, minor, patch = (int(n) for n in version.split("."))
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def sync_docs(old: str, new: str, dry_run: bool) -> list[str]:
    """Replace every `IMAGE:old` and bare `old` version mention with `new`."""
    touched = []
    for name in VERSIONED_DOCS:
        path = ROOT / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf8")
        updated = text.replace(f"{IMAGE}:{old}", f"{IMAGE}:{new}")
        updated = re.sub(rf"(?<![\d.]){re.escape(old)}(?![\d.])", new, updated)
        if updated != text:
            touched.append(name)
            if not dry_run:
                path.write_text(updated, encoding="utf8")
    return touched


def docker(args: list[str], dry_run: bool) -> None:
    printable = "docker " + " ".join(args)
    if dry_run:
        print(f"  [dry-run] {printable}")
        return
    print(f"  {printable}")
    subprocess.run(["docker", *args], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("major", "minor", "patch"), default="patch",
                        nargs="?")
    parser.add_argument("--no-cache", action="store_true", help="force a clean build")
    parser.add_argument("--no-push", action="store_true", help="build and tag only")
    parser.add_argument("--dry-run", action="store_true", help="print, change nothing")
    args = parser.parse_args()

    current = read_version()
    new = bump(current, args.part)
    print(f"version: {current} -> {new}")
    print(f"image:   {IMAGE}\n")

    core_versioned = f"{IMAGE}:{new}-core"
    full_versioned = f"{IMAGE}:{new}-full"

    print("building core (lessons 1-8)")
    build = ["build", "-f", "Dockerfile", "-t", core_versioned]
    if args.no_cache:
        build.append("--no-cache")
    build.append(".")
    docker(build, args.dry_run)

    print("\nbuilding full (adds TensorFlow, FROM core)")
    build = ["build", "-f", "Dockerfile.full",
             "--build-arg", f"CORE_IMAGE={core_versioned}", "-t", full_versioned]
    if args.no_cache:
        build.append("--no-cache")
    build.append(".")
    docker(build, args.dry_run)

    major, minor, _ = new.split(".")
    # Moving tags only; no `latest`. docker-compose.yml resolves TAI_TAG to
    # `core`, so these are the names students actually pull.
    tags = {
        core_versioned: ["core", f"{major}.{minor}-core", f"{major}-core"],
        full_versioned: ["full", f"{major}.{minor}-full", f"{major}-full"],
    }

    print("\ntagging")
    for source, moving in tags.items():
        for tag in moving:
            docker(["tag", source, f"{IMAGE}:{tag}"], args.dry_run)

    if args.no_push:
        print("\npush skipped (--no-push)")
    else:
        print("\npushing")
        for source, moving in tags.items():
            docker(["push", source], args.dry_run)
            for tag in moving:
                docker(["push", f"{IMAGE}:{tag}"], args.dry_run)

    touched = sync_docs(current, new, args.dry_run)
    if args.dry_run:
        print(f"\n[dry-run] VERSION would become {new} (left at {current})")
        if touched:
            print("[dry-run] would update: " + ", ".join(touched))
        return 0

    VERSION_FILE.write_text(new, encoding="utf8")
    print(f"\nVERSION -> {new}")
    if touched:
        print("version references updated in: " + ", ".join(touched))
    print(f'\nNext: git commit -am "Release v{new}" && git tag v{new}')
    print("Students pull `:core`; they switch .env to TAI_TAG=full before lesson 9.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
