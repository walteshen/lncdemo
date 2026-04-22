from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from lnc_demo.common import (
    add_common_model_args,
    create_udp_socket,
    make_image_id,
    payload_budget,
    choose_device,
    set_deterministic_seed,
)
from lnc_demo.model_runtime import EntropyWACNN, RawPassThrough, build_engine, load_image_as_tensor
from lnc_demo.packetization import ensure_chunk_payload_budget, packetize_bytes_chunks, packetize_int16_tensor
from lnc_demo.protocol import LossyUDPSender, SessionMeta, build_data_packet, build_end_packet, build_meta_packet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LNC source node: image -> latent packets -> relay")
    parser.add_argument("--target-host", type=str, required=True)
    parser.add_argument("--target-port", type=int, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--bind-host", type=str, default="0.0.0.0")
    parser.add_argument("--bind-port", type=int, default=0)
    add_common_model_args(parser)
    parser.add_argument("--send-interval-ms", type=float, default=1.0)
    parser.add_argument("--drop-prob", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic_seed(args.seed)
    device = choose_device(args.device)
    model, note = build_engine(args.engine, checkpoint=args.checkpoint, device=device)
    sock = create_udp_socket(args.bind_host, args.bind_port)
    sender = LossyUDPSender(sock, (args.target_host, args.target_port), drop_prob=args.drop_prob)

    bundle = load_image_as_tensor(args.image, device=device)
    image_id = make_image_id()

    if isinstance(model, RawPassThrough):
        with torch.no_grad():
            latent = model.encode(bundle.tensor)
        latent_np = latent.squeeze(0).detach().cpu().numpy().astype("int16")
        packetized = packetize_int16_tensor(latent_np, payload_budget(args.mtu), seed=args.seed)
        meta = SessionMeta(
            image_id=image_id,
            filename=Path(args.image).name,
            engine=note,
            total_packets=packetized.total_packets,
            seed=packetized.seed,
            latent_shape=packetized.shape,
            original_size=bundle.original_size,
            padded_size=bundle.padded_size,
            dtype="int16",
            coding="raw-int16",
            chunk_size=0,
        )
        packets = packetized.packets
    else:
        assert isinstance(model, EntropyWACNN)
        with torch.no_grad():
            latent = model.encode_latent(bundle.tensor)
            chunks = model.compress_latent_chunks(latent, chunk_size=args.entropy_chunk)
        ensure_chunk_payload_budget(chunks, payload_budget(args.mtu))
        packetized = packetize_bytes_chunks(chunks, seed=args.seed)
        meta = SessionMeta(
            image_id=image_id,
            filename=Path(args.image).name,
            engine=note,
            total_packets=packetized.total_packets,
            seed=packetized.seed,
            latent_shape=tuple(int(x) for x in latent.shape[1:]),
            original_size=bundle.original_size,
            padded_size=bundle.padded_size,
            dtype="entropy_bottleneck",
            coding="compressai-eb",
            chunk_size=args.entropy_chunk,
        )
        packets = packetized.packets

    meta_packet = build_meta_packet(meta)
    for _ in range(3):
        sender.sendto(meta_packet, reliable=True)
        time.sleep(0.02)

    for packet_index, payload in enumerate(packets):
        sender.sendto(build_data_packet(image_id, packet_index, meta.total_packets, payload), reliable=False)
        time.sleep(args.send_interval_ms / 1000.0)

    for _ in range(3):
        sender.sendto(build_end_packet(image_id), reliable=True)
        time.sleep(0.02)

    print(
        f"[source] sent image_id={image_id} file={Path(args.image).name} packets={meta.total_packets} "
        f"latent_shape={meta.latent_shape} coding={meta.coding} engine={note} -> {args.target_host}:{args.target_port}"
    )


if __name__ == "__main__":
    main()
