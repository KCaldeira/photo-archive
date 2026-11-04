import json

import numpy as np

from photosel.config import Config
from photosel.db import connect
from photosel.rank import run_rank


def _photo(con, sha1, ts, aest, **kw):
    row = dict(sha1=sha1, path=f"/x/{sha1}.jpg", format="JPEG", megapixels=12.0, taken_ts=ts,
               camera_model="Pixel", brightness=120.0, sharpness=200.0, sharp_center=300.0, has_thumb=1)
    zs = kw.pop("zs", None) or {"junk_screenshot": 0.01, "people": 0.99}
    row.update(kw)
    con.execute(f"INSERT INTO photos ({','.join(row)}) VALUES ({','.join('?' * len(row))})", list(row.values()))
    con.execute("INSERT INTO scores (sha1, aesthetic, iqa, iaa, zeroshot) VALUES (?,?,?,?,?)",
                (sha1, aest, 0.5, 5.0, json.dumps(zs)))


def test_rank(tmp_path):
    cfg = Config(takeout_dir=tmp_path, work_dir=tmp_path)
    con = connect(cfg.db_path)
    rng = np.random.default_rng(0)
    base = rng.normal(size=768)
    t0 = 1_700_000_000
    # Event 1: a burst of 3 near-identical shots + 1 different photo
    for i, a in enumerate([5.0, 6.5, 5.5]):
        _photo(con, f"b{i}", t0 + i * 5, a)
        e = base + 0.05 * rng.normal(size=768)
        con.execute("INSERT INTO clip VALUES (?,?)", (f"b{i}", (e / np.linalg.norm(e)).astype(np.float16).tobytes()))
    other = rng.normal(size=768)
    _photo(con, "o1", t0 + 20, 4.0)
    con.execute("INSERT INTO clip VALUES (?,?)", ("o1", (other / np.linalg.norm(other)).astype(np.float16).tobytes()))
    # Event 2 (a day later): a screenshot, a low-res photo, a favorite
    _photo(con, "shot", t0 + 86400, 3.0, format="PNG", camera_model=None)
    _photo(con, "small", t0 + 86410, 7.0, megapixels=1.0)
    _photo(con, "junk", t0 + 86420, 4.0, zs={"junk_document": 0.9, "people": 0.1})
    _photo(con, "fav", t0 + 86430, 3.0, favorited=1)
    con.commit()

    res = run_rank(cfg, {"recompute_zeroshot": False})
    r = {row["sha1"]: dict(row) for row in con.execute("SELECT * FROM rank")}
    assert r["shot"]["excluded"] == "screenshot"
    assert r["small"]["excluded"] == "low-res"
    assert r["junk"]["excluded"] == "junk:document"
    assert r["fav"]["excluded"] is None and r["fav"]["finalist"] == 1
    assert r["b0"]["burst_id"] == r["b1"]["burst_id"] == r["b2"]["burst_id"] != r["o1"]["burst_id"]
    assert r["b1"]["burst_best"] == 1 and r["b0"]["burst_best"] == 0
    assert r["b1"]["finalist"] == 1 and r["b0"]["finalist"] == 0
    assert r["b0"]["event_id"] == r["o1"]["event_id"] != r["fav"]["event_id"]
    assert res["events"] == 2
