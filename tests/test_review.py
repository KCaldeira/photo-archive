import json

import numpy as np

from photosel.db import connect
from photosel.profile import SCHEMA as PROFILE_SCHEMA
from photosel.review import MIN_FOR_MODEL, SCHEMA, Sampler, save_rating, stats


def _db(tmp_path, n=400):
    con = connect(tmp_path / "t.sqlite")
    con.executescript(PROFILE_SCHEMA)
    rng = np.random.default_rng(0)
    sha = [f"{i:040x}" for i in range(n)]
    for i, s in enumerate(sha):
        con.execute("INSERT INTO photos (sha1, path, has_thumb) VALUES (?, ?, 1)", (s, f"/x/{i}.jpg"))
        con.execute("INSERT INTO scores (sha1, aesthetic) VALUES (?, ?)", (s, float(rng.uniform(3, 7))))
        if i < 40:          # filtered out by the general ranking
            con.execute("INSERT INTO rank (sha1, excluded) VALUES (?, 'junk:object')", (s,))
            continue
        stage = 2 if i < 140 else 1
        selected = (i - 39) if i < 70 else None
        con.execute("INSERT INTO profile_scores VALUES (?,?,?,?,?,?,?,?)",
                    ("office", s, stage, float(rng.uniform()), 0.5, "people", json.dumps({}), selected))
    con.commit()
    emb = rng.normal(size=(n, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return con, dict(zip(sha, emb))


def test_batch_mixes_sources_and_never_repeats(tmp_path):
    con, emb = _db(tmp_path)
    smp = Sampler(con, "office", emb=emb, seed=1)
    first = smp.batch(20)
    assert len(first) == 20 and len({i["sha1"] for i in first}) == 20
    assert {i["source"] for i in first} == {"top", "edge", "stage1", "random", "excluded"}
    seen = {i["sha1"] for i in first}
    for _ in range(5):
        seen_now = {i["sha1"] for i in smp.batch(20)}
        assert not (seen_now & seen)
        seen |= seen_now
    verdicts = {i["source"]: i["verdict"] for i in first}
    assert verdicts["top"].startswith("selected") and verdicts["excluded"] == "filtered out: junk:object"


def test_ratings_persist_and_switch_on_the_model(tmp_path):
    con, emb = _db(tmp_path)
    log = tmp_path / "log.jsonl"
    smp = Sampler(con, "office", emb=emb, seed=2)
    items = smp.batch(MIN_FOR_MODEL + 10)
    for k, it in enumerate(items):
        save_rating(con, log, "office", it["sha1"], ["yes", "no", "maybe"][k % 3], None, it["source"])
    save_rating(con, log, "office", items[0]["sha1"], None, None, "x")          # undo one
    st = stats(con, "office")
    assert sum(st["total"].values()) == len(items) - 1
    assert len(log.read_text().splitlines()) == len(items) + 1
    assert smp.fit(con) == len(items) - 1 and smp.model_p is not None
    srcs = {i["source"] for i in Sampler(con, "office", emb=emb, seed=3).batch(20)}
    assert "uncertain" not in srcs                      # a fresh sampler has not fitted yet
    smp.rated |= {i["sha1"] for i in items}
    assert {"uncertain", "gem"} <= {i["source"] for i in smp.batch(20)}


def test_personal_is_a_no_for_this_profile(tmp_path):
    from photosel.taste import training_set

    con, emb = _db(tmp_path)
    con.executescript(SCHEMA)
    log = tmp_path / "log.jsonl"
    sha = list(emb)[40:46]
    for s in sha[:3]:
        save_rating(con, log, "office", s, "maybe", None, "top")
    for s in sha[3:]:
        save_rating(con, log, "office", s, "no", "personal", "top")
    ids, X, y = training_set(con, "office", emb)
    assert dict(zip(ids, y)) == {**{s: True for s in sha[:3]}, **{s: False for s in sha[3:]}}


def test_final_pass_shows_your_maybes_first(tmp_path):
    con, emb = _db(tmp_path)
    con.executescript(SCHEMA)
    log = tmp_path / "log.jsonl"
    sha = list(emb)[40:60]
    for k, s in enumerate(sha):
        save_rating(con, log, "office", s, "maybe" if k < 12 else "no", None, "top")
    smp = Sampler(con, "office", emb=emb, seed=4, final=True)
    first = smp.batch(20)
    assert {i["sha1"] for i in first} == set(sha[:12]) and {i["source"] for i in first} == {"final"}
    nxt = smp.batch(20)
    assert {i["source"] for i in nxt} <= {"top", "gem"} and not ({i["sha1"] for i in nxt} & set(sha))
