import json
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image

from photosel.config import Config
from photosel.ingest import run_ingest
from photosel.sheets import make_sheets


def _jpeg(path: Path, seed: int, size=(1200, 900), dt="2023:07:04 18:30:00", blur=False):
    rng = np.random.default_rng(seed)
    arr = (rng.random((size[1] // 30, size[0] // 30, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(arr).resize(size, Image.BILINEAR if blur else Image.NEAREST)
    exif = Image.Exif()
    exif[0x010F], exif[0x0110] = "Apple", "iPhone 14"
    exif.get_ifd(0x8769)[0x9003] = dt
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", exif=exif, quality=90)


def _side(path: Path, title: str, ts: int, **extra):
    d = {"title": title, "photoTakenTime": {"timestamp": str(ts)},
         "geoData": {"latitude": 0.0, "longitude": 0.0}, **extra}
    path.write_text(json.dumps(d))


def test_ingest_end_to_end(tmp_path):
    year = tmp_path / "Takeout" / "Google Photos" / "Photos from 2023"
    album = tmp_path / "Takeout" / "Google Photos" / "Beach Trip"
    _jpeg(year / "IMG_0001.JPG", 1)
    _side(year / "IMG_0001.JPG.json", "IMG_0001.JPG", 1688495400,
          people=[{"name": "Ana"}], favorited=True, geoData={"latitude": 37.4, "longitude": -122.1})
    _jpeg(year / "IMG_0001(1).JPG", 2)
    _side(year / "IMG_0001.JPG(1).json", "IMG_0001.JPG", 1688495500)
    _jpeg(year / "IMG_0002.JPG", 3, blur=True)
    _side(year / "IMG_0002.JPG.supplemental-metadata.json", "IMG_0002.JPG", 1688495600)
    _jpeg(year / "IMG_0002-edited.JPG", 4)
    long = "A_really_long_file_name_from_some_messaging_app_2023.jpg"
    _jpeg(year / long, 5, size=(640, 480))
    _side(year / ((long + ".supplemental-metadata")[:46] + ".json"), long, 1688495700)
    Image.new("RGB", (1170, 2532), "white").save(year / "Screenshot_2023.png")
    (year / "clip.mp4").write_bytes(b"\x00" * 100)
    # Same photo in an album folder
    album.mkdir(parents=True)
    (album / "IMG_0001.JPG").write_bytes((year / "IMG_0001.JPG").read_bytes())
    _side(album / "IMG_0001.JPG.json", "IMG_0001.JPG", 1688495400)
    (album / "metadata.json").write_text(json.dumps({"title": "Beach Trip"}))

    cfg = Config(takeout_dir=tmp_path / "Takeout", work_dir=tmp_path / "work", workers=2)
    cfg.work_dir.mkdir()
    res = run_ingest(cfg)
    assert res["errors"] == 0
    assert res["photos"] == 5          # 0001, 0001(1), 0002, long, screenshot (album copy deduped)
    assert res["edited_links"] == 1

    con = sqlite3.connect(cfg.db_path)
    con.row_factory = sqlite3.Row
    files = {Path(r["path"]).name + ("@album" if r["album"] else ""): r for r in con.execute("SELECT * FROM files")}
    assert files["clip.mp4"]["kind"] == "video"
    for name in ("IMG_0001.JPG", "IMG_0001(1).JPG", "IMG_0002.JPG", long, "IMG_0001.JPG@album"):
        assert files[name]["sidecar"], name
    p1 = con.execute("SELECT * FROM photos WHERE sha1=?", (files["IMG_0001.JPG"]["sha1"],)).fetchone()
    assert files["IMG_0001.JPG@album"]["sha1"] == p1["sha1"]
    assert p1["favorited"] == 1 and json.loads(p1["people"]) == ["Ana"]
    assert json.loads(p1["albums"]) == ["Beach Trip"]
    assert abs(p1["lat"] - 37.4) < 1e-6 and p1["date_source"] == "sidecar"
    assert p1["camera_model"] == "iPhone 14" and p1["taken_local"] == "2023-07-04T18:30:00"
    p2 = con.execute("SELECT * FROM photos WHERE path LIKE '%IMG_0002.JPG'").fetchone()
    assert p2["edited_path"].endswith("IMG_0002-edited.JPG")
    assert p2["sharpness"] < p1["sharpness"]   # blurred image scores lower
    shot = con.execute("SELECT * FROM photos WHERE path LIKE '%Screenshot%'").fetchone()
    assert shot["camera_model"] is None and shot["date_source"] == "mtime" and shot["clip_light"] > 0.99
    assert all(cfg.thumb_path(r["sha1"]).exists() for r in con.execute("SELECT sha1 FROM photos"))

    # Incremental: nothing new to do on a second run
    assert run_ingest(cfg)["processed"] == 0

    rows = [dict(r) for r in con.execute("SELECT sha1, path FROM photos")]
    sheets = make_sheets(cfg, rows, "t")
    assert len(sheets) == 1 and sheets[0].exists()
