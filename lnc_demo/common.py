from __future__ import annotations

import argparse
import os
import random
import socket
import time
from pathlib import Path

import numpy as np
import torch


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=["wacnn-demo", "raw"], default="wacnn-demo")
    parser.add_argument("--checkpoint", type=str, default=None, help="Training checkpoint (.pth) matching the selected model")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--mtu", type=int, default=1472, help="UDP payload budget before IP fragmentation")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--entropy-chunk",
        type=int,
        default=16,
        help="Number of flattened latent positions per entropy-coded packet chunk for compressai mode",
    )


def add_common_network_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bind-host", type=str, default="0.0.0.0")
    parser.add_argument("--bind-port", type=int, required=True)
    parser.add_argument("--drop-prob", type=float, default=0.0, help="Local packet drop emulation on outgoing packets")


def create_udp_socket(bind_host: str, bind_port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind_host, bind_port))
    return sock


def payload_budget(mtu: int, header_bytes: int = 32) -> int:
    # Application-safe budget.
    return max(64, mtu - header_bytes)


def choose_device(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_image_id() -> int:
    return int(time.time() * 1000) & 0xFFFFFFFF


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
