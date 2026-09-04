"""Fetch the Commander Spellbook bulk export, politely.

Commander Spellbook's API docs (backend repo, ``docs/api.md``) say:

    Do not use the HTTP API to export the whole dataset. Every variant, together
    with the variant aliases, is published as a single JSON document on S3,
    refreshed periodically:
        https://json.commanderspellbook.com/variants.json.gz  (prefer this one)
        https://json.commanderspellbook.com/variants.json

    Name your service in the User-Agent header. Please credit us and link back
    to https://commanderspellbook.com/ where applicable.

This module downloads the gzipped document once, keeps it on disk, and only
re-downloads when the server reports a newer file (ETag / Last-Modified).
"""

from __future__ import annotations

import argparse
import email.utils
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BULK_VARIANTS_GZIP_URL = "https://json.commanderspellbook.com/variants.json.gz"
BULK_VARIANTS_URL = "https://json.commanderspellbook.com/variants.json"
DEFAULT_USER_AGENT = "mtg-combos-subgraph/0.1 (+https://github.com/dubiousllama/mtg-combos-subgraph)"


def _meta_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ".meta.json")


def download_bulk(target: Path, url: str = BULK_VARIANTS_GZIP_URL, user_agent: str = DEFAULT_USER_AGENT, force: bool = False) -> bool:
    """Download ``url`` to ``target`` unless the cached copy is still current.

    Returns True if a new file was written, False if the cached copy was kept.
    Uses conditional requests so a no-op refresh costs the server one 304.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    meta_path = _meta_path(target)
    headers = {"User-Agent": user_agent, "Accept-Encoding": "identity"}
    if target.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        if meta.get("etag"):
            headers["If-None-Match"] = meta["etag"]
        if meta.get("last_modified"):
            headers["If-Modified-Since"] = meta["last_modified"]
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        if error.code == 304:
            print(f"{target} is up to date (server returned 304)", file=sys.stderr)
            return False
        raise
    with response:
        tmp = target.with_suffix(target.suffix + ".part")
        total = 0
        with tmp.open("wb") as f:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
        os.replace(tmp, target)
        meta = {
            "url": url,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "downloaded_at": email.utils.formatdate(usegmt=True),
            "bytes": total,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
    print(f"downloaded {total:,} bytes to {target}", file=sys.stderr)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/variants.json.gz"))
    parser.add_argument("--url", default=BULK_VARIANTS_GZIP_URL)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--force", action="store_true", help="ignore the cached copy")
    args = parser.parse_args(argv)
    download_bulk(args.out, url=args.url, user_agent=args.user_agent, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
