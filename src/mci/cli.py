"""Command line: `mci fetch | train | serve | learner | simulate | loadtest-local | drift`."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

DATA = Path(os.environ.get("MCI_DATA", "data/spotify_songs.csv"))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(prog="mci")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download and verify the dataset")
    tr = sub.add_parser("train", help="baselines, variants, final model, bundle, gate, MLflow registry")
    tr.add_argument("--out", default="bundles")
    tr.add_argument("--quick", action="store_true")
    tr.add_argument("--variant", action="append")
    tr.add_argument("--report", default="reports/training.json")
    sv = sub.add_parser("serve")
    sv.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    sv.add_argument("--workers", type=int, default=int(os.environ.get("WEB_CONCURRENCY", "1")))
    sub.add_parser("learner", help="consume listening events and update the online model")
    sm = sub.add_parser("simulate", help="offline benchmark of the online learner against simulated listeners")
    sm.add_argument("--hours", type=int, default=72)
    sm.add_argument("--out", default="reports/online_learning.json")
    sm.add_argument("--bundle")
    sm.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args(argv)

    if a.cmd == "fetch":
        from mci import data

        print(data.fetch(DATA))
    elif a.cmd == "train":
        from mci import train

        report = train.run(DATA, Path(a.out), quick=a.quick, variants=a.variant)
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps({k: report[k] for k in ("selected", "playlist_candidates", "popularity_candidates", "gate", "bundle")}, indent=2, default=str))
    elif a.cmd == "serve":
        import uvicorn

        uvicorn.run("mci.serving.app:create_app", factory=True, host="0.0.0.0", port=a.port, workers=a.workers, proxy_headers=True)
    elif a.cmd == "learner":
        from mci.online import run_learner

        run_learner()
    elif a.cmd == "simulate":
        from mci.simulate import run

        r = run(DATA, Path(a.bundle) if a.bundle else None, hours=a.hours, seeds=a.seeds)
        Path(a.out).write_text(json.dumps(r, indent=2))
        print(json.dumps(r["summary"], indent=2))


if __name__ == "__main__":
    main()
