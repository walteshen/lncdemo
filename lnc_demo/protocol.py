from __future__ import annotations

import json
import random
import socket
import struct
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

MAGIC = b"LNC1"
VERSION = 1

KIND_META = 1
KIND_DATA = 2
KIND_END = 3

META_HEADER = struct.Struct("!4sBBIH")
DATA_HEADER = struct.Struct("!4sBBIIIH")


@dataclass
class SessionMeta:
    image_id: int
    filename: str
    engine: str
    total_packets: int
    seed: int
    latent_shape: Tuple[int, int, int]
    original_size: Tuple[int, int]
    padded_size: Tuple[int, int]
    dtype: str = "int16"
    coding: str = "raw-int16"
    chunk_size: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "image_id": self.image_id,
            "filename": self.filename,
            "engine": self.engine,
            "total_packets": self.total_packets,
            "seed": self.seed,
            "latent_shape": list(self.latent_shape),
            "original_size": list(self.original_size),
            "padded_size": list(self.padded_size),
            "dtype": self.dtype,
            "coding": self.coding,
            "chunk_size": self.chunk_size,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "SessionMeta":
        return cls(
            image_id=int(payload["image_id"]),
            filename=str(payload.get("filename", "image")),
            engine=str(payload.get("engine", "wacnn-demo")),
            total_packets=int(payload["total_packets"]),
            seed=int(payload["seed"]),
            latent_shape=tuple(int(x) for x in payload["latent_shape"]),  # type: ignore[arg-type]
            original_size=tuple(int(x) for x in payload["original_size"]),  # type: ignore[arg-type]
            padded_size=tuple(int(x) for x in payload["padded_size"]),  # type: ignore[arg-type]
            dtype=str(payload.get("dtype", "int16")),
            coding=str(payload.get("coding", "raw-int16")),
            chunk_size=int(payload.get("chunk_size", 0)),
        )


class LossyUDPSender:
    def __init__(self, sock: socket.socket, target: Tuple[str, int], drop_prob: float = 0.0) -> None:
        self.sock = sock
        self.target = target
        self.drop_prob = max(0.0, min(1.0, drop_prob))

    def sendto(self, payload: bytes, reliable: bool = False) -> None:
        if (not reliable) and random.random() < self.drop_prob:
            return
        self.sock.sendto(payload, self.target)


def build_meta_packet(meta: SessionMeta) -> bytes:
    payload = json.dumps(meta.to_dict(), separators=(",", ":")).encode("utf-8")
    return META_HEADER.pack(MAGIC, VERSION, KIND_META, meta.image_id, len(payload)) + payload


def build_end_packet(image_id: int) -> bytes:
    payload = b"{}"
    return META_HEADER.pack(MAGIC, VERSION, KIND_END, image_id, len(payload)) + payload


def build_data_packet(image_id: int, packet_index: int, total_packets: int, payload: bytes) -> bytes:
    return DATA_HEADER.pack(
        MAGIC,
        VERSION,
        KIND_DATA,
        image_id,
        packet_index,
        total_packets,
        len(payload),
    ) + payload


@dataclass
class ParsedPacket:
    kind: int
    image_id: int
    packet_index: Optional[int] = None
    total_packets: Optional[int] = None
    payload: bytes = b""


def parse_packet(packet: bytes) -> Optional[ParsedPacket]:
    if len(packet) < META_HEADER.size:
        return None
    magic = packet[:4]
    if magic != MAGIC:
        return None
    version = packet[4]
    kind = packet[5]
    if version != VERSION:
        return None
    if kind in (KIND_META, KIND_END):
        magic, version, kind, image_id, payload_len = META_HEADER.unpack(packet[: META_HEADER.size])
        return ParsedPacket(kind=kind, image_id=image_id, payload=packet[META_HEADER.size : META_HEADER.size + payload_len])
    if kind == KIND_DATA:
        magic, version, kind, image_id, packet_index, total_packets, payload_len = DATA_HEADER.unpack(packet[: DATA_HEADER.size])
        return ParsedPacket(
            kind=kind,
            image_id=image_id,
            packet_index=packet_index,
            total_packets=total_packets,
            payload=packet[DATA_HEADER.size : DATA_HEADER.size + payload_len],
        )
    return None


@dataclass
class ReceivedSession:
    meta: SessionMeta
    packets: Dict[int, bytes]
    end_received: bool
    sender_addr: Tuple[str, int]


class SessionReceiver:
    def __init__(self, sock: socket.socket, timeout_s: float = 5.0, idle_timeout_s: float = 1.0) -> None:
        self.sock = sock
        self.timeout_s = timeout_s
        self.idle_timeout_s = idle_timeout_s

    def receive_one_session(self) -> Optional[ReceivedSession]:
        self.sock.settimeout(self.timeout_s)
        start = time.time()
        meta: Optional[SessionMeta] = None
        packets: Dict[int, bytes] = {}
        end_received = False
        sender_addr: Optional[Tuple[str, int]] = None
        last_activity = time.time()

        while True:
            if meta is not None and (end_received or (time.time() - last_activity) > self.idle_timeout_s):
                return ReceivedSession(meta=meta, packets=packets, end_received=end_received, sender_addr=sender_addr or ("", 0))
            if meta is None and (time.time() - start) > self.timeout_s:
                return None

            try:
                packet, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                if meta is not None:
                    return ReceivedSession(meta=meta, packets=packets, end_received=end_received, sender_addr=sender_addr or ("", 0))
                return None

            parsed = parse_packet(packet)
            if parsed is None:
                continue
            sender_addr = addr
            last_activity = time.time()

            if parsed.kind == KIND_META:
                meta = SessionMeta.from_dict(json.loads(parsed.payload.decode("utf-8")))
            elif parsed.kind == KIND_DATA and meta is not None and parsed.packet_index is not None:
                packets[parsed.packet_index] = parsed.payload
            elif parsed.kind == KIND_END and meta is not None:
                end_received = True
