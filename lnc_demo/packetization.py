from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PacketizedTensor:
    packets: List[bytes]
    total_packets: int
    seed: int
    shape: Tuple[int, int, int]


@dataclass
class PacketizedBytes:
    packets: List[bytes]
    total_packets: int
    seed: int
    num_chunks: int


def _compute_num_packets(total_elements: int, bytes_per_element: int, payload_bytes: int) -> int:
    elements_per_packet = max(1, payload_bytes // bytes_per_element)
    return int(math.ceil(total_elements / elements_per_packet))


def packetize_int16_tensor(tensor: np.ndarray, payload_bytes: int, seed: int) -> PacketizedTensor:
    if tensor.dtype != np.int16:
        raise ValueError("tensor must be int16")
    flat = tensor.reshape(-1)
    total_elements = int(flat.size)
    total_packets = _compute_num_packets(total_elements, bytes_per_element=2, payload_bytes=payload_bytes)
    rng = np.random.default_rng(seed)
    order = rng.permutation(total_elements)
    packets: List[bytes] = []
    for packet_index in range(total_packets):
        idx = order[packet_index::total_packets]
        values = flat[idx]
        packets.append(values.astype(np.int16, copy=False).tobytes())
    return PacketizedTensor(packets=packets, total_packets=total_packets, seed=seed, shape=tuple(int(x) for x in tensor.shape))


def depacketize_int16_tensor(
    payloads: Dict[int, bytes],
    shape: Tuple[int, int, int],
    total_packets: int,
    seed: int,
) -> np.ndarray:
    total_elements = int(np.prod(shape))
    out = np.zeros(total_elements, dtype=np.int16)
    rng = np.random.default_rng(seed)
    order = rng.permutation(total_elements)
    for packet_index, payload in payloads.items():
        if not (0 <= packet_index < total_packets):
            continue
        idx = order[packet_index::total_packets]
        values = np.frombuffer(payload, dtype=np.int16, count=len(idx))
        out[idx[: len(values)]] = values[: len(idx)]
    return out.reshape(shape)


def ensure_chunk_payload_budget(chunks: List[bytes], payload_bytes: int) -> None:
    if not chunks:
        return
    longest = max(len(chunk) for chunk in chunks)
    if longest > payload_bytes:
        raise ValueError(
            f"One entropy-coded chunk is {longest} bytes, which exceeds the UDP payload budget {payload_bytes}. "
            "Decrease --entropy-chunk or increase --mtu."
        )


def packetize_bytes_chunks(chunks: List[bytes], seed: int) -> PacketizedBytes:
    num_chunks = len(chunks)
    if num_chunks == 0:
        return PacketizedBytes(packets=[], total_packets=0, seed=seed, num_chunks=0)
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_chunks)
    packets = [chunks[int(chunk_idx)] for chunk_idx in order]
    return PacketizedBytes(packets=packets, total_packets=num_chunks, seed=seed, num_chunks=num_chunks)


def depacketize_bytes_chunks(
    payloads: Dict[int, bytes],
    num_chunks: int,
    seed: int,
) -> List[Optional[bytes]]:
    if num_chunks <= 0:
        return []
    ordered: List[Optional[bytes]] = [None] * num_chunks
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(num_chunks)
    inverse = np.empty(num_chunks, dtype=np.int64)
    inverse[permutation] = np.arange(num_chunks)
    for packet_index, payload in payloads.items():
        if not (0 <= packet_index < num_chunks):
            continue
        chunk_index = int(permutation[packet_index])
        ordered[chunk_index] = payload
    return ordered
