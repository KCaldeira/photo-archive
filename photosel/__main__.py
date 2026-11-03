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

    sub.add_parser("stats", help="summarize the database")

    p = sub.add_parser("sheet", help="make numbered contact sheets")
    p.add_argument("--name", default="sample")
    p.add_argument("--sample", type=int, help="random sample of N photos")
    p.add_argument("--where", default="1", help="SQL WHERE clause on photos")
    p.add_argument("--order", default="RANDOM()", help="SQL ORDER BY on photos")
    p.add_argument("--limit", type=int, default=100)

    a = ap.parse_args(argv)
    cfg = load_config(a.config, takeout_dir=a.takeout_dir, work_dir=a.work_dir,
                      workers=getattr(a, "workers", None))

    if a.cmd == "ingest":
        from .ingest import run_ingest
        print(json.dumps(run_ingest(cfg, limit=a.limit), indent=2))
    elif a.cmd == "stats":
        from .report import print_stats
        print_stats(cfg)
    elif a.cmd == "sheet":
        from .db import connect
        from .sheets import make_sheets
        con = connect(cfg.db_path)
        n = a.sample or a.limit
        rows = [dict(r) for r in con.execute(
            f"SELECT sha1, path, taken_at, megapixels, sharpness FROM photos "
            f"WHERE has_thumb=1 AND ({a.where}) ORDER BY {a.order} LIMIT ?", (n,))]
        caps = [f"{(r['taken_at'] or '')[:10]} {r['megapixels'] or 0:.0f}MP s{r['sharpness'] or 0:.0f}" for r in rows]
        for s in make_sheets(cfg, rows, a.name, caps):
            print(s)


if __name__ == "__main__":
    main()
