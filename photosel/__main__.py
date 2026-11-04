"""Command line: python -m photosel <command>."""
from __future__ import annotations

import argparse
import json

from .config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="photosel")
    ap.add_argument("--config", help="config file (default: repo config.yaml)")
    ap.add_argument("--takeout-dir", help="override takeout_dir")
    ap.add_argument("--work-dir", help="override work_dir")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="index Takeout, make thumbnails and cheap stats")
    p.add_argument("--limit", type=int, help="process at most N new images (for a quick trial)")
    p.add_argument("--workers", type=int)

    p = sub.add_parser("embed", help="GPU: CLIP embeddings, aesthetic/quality scores, zero-shot labels")
    p.add_argument("--limit", type=int)
    p.add_argument("--skip-iqa", action="store_true")

    p = sub.add_parser("detect", help="GPU: person detection on thumbnails")
    p.add_argument("--limit", type=int)

    p = sub.add_parser("structure", help="coarse-scale structure metrics from thumbnails (CPU)")
    p.add_argument("--workers", type=int, default=4)

    p = sub.add_parser("profile", help="purpose-specific scoring, e.g. 'profile office'")
    p.add_argument("name")
    p.add_argument("--stage1-only", action="store_true", help="skip the expensive shortlist metrics")

    sub.add_parser("rank", help="filters, bursts, events, composite score, finalists (fast)")

    sub.add_parser("stats", help="summarize the database")

    p = sub.add_parser("sheet", help="make numbered contact sheets")
    p.add_argument("--name", default="sample")
    p.add_argument("--sample", type=int, help="random sample of N photos")
    p.add_argument("--where", default="1", help="SQL WHERE clause on photos")
    p.add_argument("--order", default="RANDOM()", help="SQL ORDER BY on photos")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--profile", help="caption with this profile's score and weakest metric")

    a = ap.parse_args(argv)
    cfg = load_config(a.config, takeout_dir=a.takeout_dir, work_dir=a.work_dir,
                      workers=getattr(a, "workers", None))

    if a.cmd == "ingest":
        from .ingest import run_ingest
        print(json.dumps(run_ingest(cfg, limit=a.limit), indent=2))
    elif a.cmd == "embed":
        from .embed import run_clip, run_iqa
        print("clip:", run_clip(cfg, limit=a.limit))
        if not a.skip_iqa:
            print("iqa:", run_iqa(cfg, limit=a.limit))
    elif a.cmd == "detect":
        from .detect import run_detect
        print("detect:", run_detect(cfg, limit=a.limit))
    elif a.cmd == "structure":
        from .structure import run_coarse
        print("structure:", run_coarse(cfg, workers=a.workers))
    elif a.cmd == "profile":
        from .profile import run_profile
        print(json.dumps(run_profile(cfg, a.name, stage2=not a.stage1_only), indent=2, default=str))
    elif a.cmd == "rank":
        from .rank import run_rank
        print(json.dumps(run_rank(cfg, cfg.rank), indent=2))
    elif a.cmd == "stats":
        from .report import print_stats
        print_stats(cfg)
    elif a.cmd == "sheet":
        from .db import connect
        from .sheets import make_sheets
        con = connect(cfg.db_path)
        n = a.sample or a.limit
        if a.profile:
            rows = [dict(r) for r in con.execute(
                f"SELECT p.sha1, p.path, p.taken_at, f.score, f.p_topk, f.weakest, f.d, f.selected "
                f"FROM photos p JOIN profile_scores f ON f.sha1=p.sha1 AND f.profile=? "
                f"WHERE ({a.where}) ORDER BY {a.order} LIMIT ?", (a.profile, n))]
            caps = [f"{r['score']:.2f} p{r['p_topk']:.2f} -{r['weakest']} {json.loads(r['d'])[r['weakest']]:.2f}"
                    for r in rows]
            for s in make_sheets(cfg, rows, a.name, caps):
                print(s)
            return
        rows = [dict(r) for r in con.execute(
            f"SELECT p.sha1, p.path, p.taken_at, p.megapixels, s.aesthetic, r.composite, r.excluded "
            f"FROM photos p LEFT JOIN scores s USING(sha1) LEFT JOIN rank r USING(sha1) "
            f"WHERE p.has_thumb=1 AND ({a.where}) ORDER BY {a.order} LIMIT ?", (n,))]

        def cap(r):
            c = f"{(r['taken_at'] or '')[:10]} {r['megapixels'] or 0:.0f}MP"
            if r["composite"] is not None:
                c += f" c{r['composite']:+.1f} a{r['aesthetic'] or 0:.1f}"
            if r["excluded"]:
                c = f"{r['excluded']} {c[5:10]}"
            return c
        caps = [cap(r) for r in rows]
        for s in make_sheets(cfg, rows, a.name, caps):
            print(s)


if __name__ == "__main__":
    main()
