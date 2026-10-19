"""Fetch and unzip a USLM release point from uscode.house.gov (§5.1).

Only the official OLRC site is used; if a download fails, the caller stops
and reports, rather than switching to a mirror.
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://uscode.house.gov/download/"
DOWNLOAD_PAGE = "https://uscode.house.gov/download.html"  # moved from /download/download.html
DEFAULT_RELEASE = "119-121"
_ALL_RE = re.compile(r"releasepoints/us/pl/(\d+)/(\d+[a-z]*)/xml_uscAll@(\d+-\d+[a-z]*)\.zip")
USER_AGENT = "negconj-research/0.1 (statutory corpus study)"


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def current_release_point() -> str | None:
    """The release point of the all-titles XML zip linked from download.html."""
    html = _get(DOWNLOAD_PAGE).decode("utf-8", "replace")
    m = _ALL_RE.search(html)
    return m.group(3) if m else None


def zip_url(release: str, title: str = "All") -> str:
    congress, law = release.split("-", 1)
    return f"{BASE}releasepoints/us/pl/{congress}/{law}/xml_usc{title}@{release}.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> Path:
    """Download to dest (via a temp name); skip if dest exists."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    with zipfile.ZipFile(tmp) as z:  # fail now on a truncated or non-zip body
        bad = z.testzip()
        if bad is not None:
            raise IOError(f"corrupt member {bad} in {url}")
    tmp.replace(dest)
    return dest


def unzip_xml(zpath: Path, out_dir: Path) -> list[Path]:
    """Extract the usc*.xml members (flattened); returns their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if not re.fullmatch(r"usc\d+a?\.xml", name):
                continue
            target = out_dir / name
            if not target.exists() or target.stat().st_size != info.file_size:
                tmp = target.with_name(name + ".part")
                with z.open(info) as src, open(tmp, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                tmp.replace(target)
            paths.append(target)
    return sorted(paths)
