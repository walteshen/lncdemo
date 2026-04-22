from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convenience runner for a single-machine 3-terminal LNC demo")
    parser.add_argument("--image", required=True)
    parser.add_argument("--engine", choices=["wacnn-demo", "raw"], default="raw")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--entropy-chunk", type=int, default=16)
    parser.add_argument("--src-drop", type=float, default=0.1)
    parser.add_argument("--relay-drop", type=float, default=0.1)
    parser.add_argument("--outdir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    base = [sys.executable]

    common = ["--engine", args.engine, "--device", args.device, "--entropy-chunk", str(args.entropy_chunk)]
    dest_cmd = base + [
        str(root / "destination.py"),
        "--bind-port", "9202",
        *common,
        "--output-dir", args.outdir,
    ]
    relay_cmd = base + [
        str(root / "relay.py"),
        "--bind-port", "9201",
        "--target-host", "127.0.0.1",
        "--target-port", "9202",
        *common,
        "--drop-prob", str(args.relay_drop),
    ]
    src_cmd = base + [
        str(root / "source.py"),
        "--target-host", "127.0.0.1",
        "--target-port", "9201",
        "--image", args.image,
        *common,
        "--drop-prob", str(args.src_drop),
    ]
    if args.checkpoint:
        dest_cmd += ["--checkpoint", args.checkpoint]
        relay_cmd += ["--checkpoint", args.checkpoint]
        src_cmd += ["--checkpoint", args.checkpoint]

    dest = subprocess.Popen(dest_cmd, cwd=root, env=env)
    relay = subprocess.Popen(relay_cmd, cwd=root, env=env)
    try:
        time.sleep(1.0)
        subprocess.run(src_cmd, cwd=root, env=env, check=True)
        time.sleep(2.0)
    finally:
        relay.terminate()
        dest.terminate()


if __name__ == "__main__":
    main()
