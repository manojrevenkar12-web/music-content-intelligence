"""Serving benchmark: what batching and int8 buy, measured on this machine.

    python scripts/bench_serving.py --bundle bundles/current --out reports/serving.json

Part 1 (engine): time per model call at batch sizes 1-256, int8 vs fp32, one ONNX thread.
Part 2 (API): starts the real server (one uvicorn worker, one ONNX thread: one pod's shape) for
each configuration in CONFIGS and load-tests it with scripts/loadtest.py. Latency includes HTTP,
validation and featurisation, and the load generator shares the machine's CPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def engine_bench(bundle: Path) -> dict:
    import random

    from loadtest import track

    from mci.serving.engine import Engine

    rng = random.Random(0)
    items = [track(rng) for _ in range(256)]
    out = {}
    for prec in ("int8", "fp32"):
        e = Engine.load(bundle, precision=prec, threads=1)
        out[prec] = {}
        for bs in (1, 8, 32, 64, 256):
            x = e.featurise(items[:bs])
            for _ in range(20):
                e.run(x)
            n = max(20, 2000 // bs)
            t0 = time.perf_counter()
            for _ in range(n):
                e.run(x)
            ms = (time.perf_counter() - t0) / n * 1000
            out[prec][str(bs)] = {"ms_per_call": round(ms, 3), "us_per_track": round(ms * 1000 / bs, 1)}
    return out


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


CONFIGS = [
    # name, precision, max batch, wait ms, clients, tracks per request
    ("saturation: int8, no batching", "int8", 1, 0, 32, 1),
    ("saturation: int8, opportunistic batching (default)", "int8", 64, 0, 32, 1),
    ("saturation: int8, 2 ms batching window", "int8", 64, 2, 32, 1),
    ("saturation: fp32, opportunistic batching", "fp32", 64, 0, 32, 1),
    ("moderate load: int8, default, 4 clients", "int8", 64, 0, 4, 1),
    ("bulk catalogue scoring: int8, 64 tracks per request, 4 clients", "int8", 256, 0, 4, 64),
]


def api_bench(bundle: Path, seconds: float) -> dict:
    out = {}
    for name, prec, max_batch, wait, users, per_req in CONFIGS:
        port = free_port()
        env = {**os.environ, "MCI_BUNDLE": str(bundle), "MCI_PRECISION": prec, "MCI_THREADS": "1", "MCI_MAX_BATCH": str(max_batch),
               "MCI_BATCH_WAIT_MS": str(wait), "LOG_LEVEL": "WARNING"}  # fmt: skip
        env.pop("REDIS_URL", None)
        srv = subprocess.Popen(
            [sys.executable, "-m", "mci.cli", "serve", "--port", str(port), "--workers", "1"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            for _ in range(100):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/readyz", timeout=1)
                    break
                except OSError:
                    time.sleep(0.2)
            rep = Path(f"/tmp/mci-bench-{len(out)}.json")
            cmd = [sys.executable, str(Path(__file__).parent / "loadtest.py"), "--base", f"http://127.0.0.1:{port}", "--users", str(users), "--seconds", str(seconds),
                   "--vcpus", "1", "--tracks-per-request", str(per_req), "--out", str(rep)]  # fmt: skip
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            r = json.loads(rep.read_text())
            out[name] = {k: r[k] for k in ("users", "tracks_per_request", "throughput_rps", "tracks_per_second", "latency_ms", "errors", "cost")}
        finally:
            srv.terminate()
            srv.wait(10)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="bundles/current")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--out", default="reports/serving.json")
    a = ap.parse_args()
    bundle = Path(a.bundle).resolve()
    rep = {
        "machine": f"{os.cpu_count()} vCPU cloud container; server: 1 uvicorn worker, 1 ONNX thread (one pod); load generator on the same host",
        "model": json.loads((bundle / "manifest.json").read_text())["version"],
        "engine": engine_bench(bundle),
        "api": api_bench(bundle, a.seconds),
        "load": f"closed-loop clients (each sends its next request when the last is answered), {a.seconds:.0f} s per configuration",
    }
    Path(a.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    sys.exit(main())
