"""Google Takeout quirks: classifying files, matching JSON sidecars, albums.

Sidecar naming seen in Takeout exports (oldest to newest formats):
    IMG_1234.JPG            -> IMG_1234.JPG.json
    IMG_1234(1).JPG         -> IMG_1234.JPG(1).json        (name clash in a folder)
    IMG_1234-edited.JPG     -> uses the original's sidecar
    IMG_1234.JPG            -> IMG_1234.JPG.supplemental-metadata.json   (2024+)
    long names              -> whole sidecar name cut to 51 chars, e.g.
                               IMG_1234.JPG.supplemental-met.json or
                               Screenshot_2023-01-01-12-00-00-123_com.exampl.json
    sometimes               -> IMG_1234.json (media extension dropped)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".heic", ".heif", ".png", ".webp", ".tif", ".tiff", ".gif", ".bmp"}
RAW_EXT = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf"}
VIDEO_EXT = {".mp", ".mp~2",  # Pixel motion-photo clips
             ".mp4", ".mov", ".m4v", ".avi", ".3gp", ".mkv", ".mts", ".m2ts", ".wmv", ".mpg"}

SUPPLEMENTAL = "supplemental-metadata"
EDITED_SUFFIXES = ("-edited", "-bearbeitet", "-modifié", "-editado", "-modificato")
NON_ALBUM_FOLDERS = {"archive", "trash", "bin", "failed videos", "google photos", "takeout"}
YEAR_FOLDER = re.compile(r"^photos from \d{4}$", re.I)
DUP_IN_NAME = re.compile(r"^(?P<stem>.*)\((?P<n>\d+)\)(?P<ext>\.[^.]+)$")
DUP_AT_END = re.compile(r"^(?P<key>.*)\((?P<n>\d+)\)$")

# Sidecar names are truncated so that "<key>.json" is at most this long.
MAX_SIDECAR_NAME = 51


def classify(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in RAW_EXT:
        return "raw"
    if ext in VIDEO_EXT:
        return "video"
    if ext == ".json":
        return "json"
    return "other"


def album_for_folder(folder: str) -> str | None:
    """Named album for a Takeout folder, or None for year/system folders."""
    f = folder.strip().lower()
    if YEAR_FOLDER.match(f) or f in NON_ALBUM_FOLDERS:
        return None
    return folder


@dataclass
class MediaName:
    name: str          # actual filename
    original: str      # filename of the unedited, un-numbered photo
    n: int             # duplicate index from "(n)", 0 if none
    is_edited: bool


def parse_media_name(name: str) -> MediaName:
    n = 0
    base = name
    m = DUP_IN_NAME.match(name)
    if m:
        n = int(m["n"])
        base = m["stem"] + m["ext"]
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    is_edited = False
    for suf in EDITED_SUFFIXES:
        if stem.lower().endswith(suf):
            stem = stem[: -len(suf)]
            is_edited = True
            break
    original = f"{stem}.{ext}" if ext else stem
    return MediaName(name=name, original=original, n=n, is_edited=is_edited)


def sidecar_key(json_name: str) -> tuple[str, int]:
    """(normalized media name it describes, duplicate index) for a sidecar file name."""
    key = json_name[:-5] if json_name.lower().endswith(".json") else json_name
    n = 0
    m = DUP_AT_END.match(key)
    if m:
        key, n = m["key"], int(m["n"])
    # Strip a (possibly truncated) ".supplemental-metadata" suffix.
    head, dot, tail = key.rpartition(".")
    # Only when what remains still has a media extension (IMG_1234.JPG.supp -> IMG_1234.JPG).
    if dot and tail and "." in head and SUPPLEMENTAL.startswith(tail.lower()):
        key = head
    return key, n


class SidecarIndex:
    """Matches media files in one folder to their JSON sidecars."""

    def __init__(self, json_paths: list[Path]):
        self.by_key: dict[tuple[str, int], Path] = {}
        self.truncated: list[tuple[str, int, Path]] = []
        for p in json_paths:
            if p.name.lower() in ("metadata.json", "print-subscriptions.json", "shared_album_comments.json",
                                  "user-generated-memory-titles.json"):
                continue
            key, n = sidecar_key(p.name)
            self.by_key.setdefault((key.lower(), n), p)
            if len(p.name) >= MAX_SIDECAR_NAME - 6:
                self.truncated.append((key.lower(), n, p))

    def match(self, media_name: str) -> Path | None:
        mn = parse_media_name(media_name)
        name_l = mn.name.lower()
        orig_l = mn.original.lower()
        orig_stem_l = orig_l.rpartition(".")[0] or orig_l
        for cand in ((name_l, 0), (orig_l, mn.n), (orig_stem_l, mn.n), (orig_l, 0)):
            if cand in self.by_key:
                return self.by_key[cand]
        # Truncated sidecar names: longest key that is a prefix of the original name.
        best = None
        for key, n, p in self.truncated:
            if n == mn.n and (orig_l.startswith(key) or name_l.startswith(key)):
                if best is None or len(key) > len(best[0]):
                    best = (key, p)
        return best[1] if best else None


def read_sidecar(path: Path) -> dict:
    """Fields we use from a Takeout photo sidecar."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict = {}
    ts = (d.get("photoTakenTime") or {}).get("timestamp")
    if ts and str(ts).isdigit() and int(ts) > 0:
        out["taken_ts"] = int(ts)
    for key in ("geoData", "geoDataExif"):
        g = d.get(key) or {}
        lat, lon = g.get("latitude"), g.get("longitude")
        if lat and lon and (abs(lat) > 1e-6 or abs(lon) > 1e-6):
            out["lat"], out["lon"] = float(lat), float(lon)
            break
    people = [p.get("name") for p in d.get("people") or [] if p.get("name")]
    if people:
        out["people"] = people
    if "favorited" in d:
        out["favorited"] = 1 if d["favorited"] else 0
    if d.get("description"):
        out["description"] = d["description"]
    if d.get("title"):
        out["title"] = d["title"]
    return out
