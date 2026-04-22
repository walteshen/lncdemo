from __future__ import annotations

import argparse
import time

import torch

from lnc_demo.common import (
    add_common_model_args,
    add_common_network_args,
    create_udp_socket,
    payload_budget,
    choose_device,
    set_deterministic_seed,
)
from lnc_demo.model_runtime import EntropyWACNN, RawPassThrough, build_engine
from lnc_demo.packetization import (
    depacketize_bytes_chunks,
    depacketize_int16_tensor,
    ensure_chunk_payload_budget,
    packetize_bytes_chunks,
    packetize_int16_tensor,
)
from lnc_demo.protocol import LossyUDPSender, SessionMeta, SessionReceiver, build_data_packet, build_end_packet, build_meta_packet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LNC relay node: partial latent -> recoding -> forward")
    add_common_network_args(parser)
    parser.add_argument("--target-host", type=str, required=True)
    parser.add_argument("--target-port", type=int, required=True)
    add_common_model_args(parser)
    parser.add_argument("--recv-timeout-s", type=float, default=30.0)
    parser.add_argument("--idle-timeout-s", type=float, default=1.0)
    parser.add_argument("--send-interval-ms", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic_seed(args.seed)
    device = choose_device(args.device)
    model, note = build_engine(args.engine, checkpoint=args.checkpoint, device=device)
    sock = create_udp_socket(args.bind_host, args.bind_port)
    sender = LossyUDPSender(sock, (args.target_host, args.target_port), drop_prob=args.drop_prob)
    receiver = SessionReceiver(sock, timeout_s=args.recv_timeout_s, idle_timeout_s=args.idle_timeout_s)

    print(f"[relay] listening on {args.bind_host}:{args.bind_port}, forwarding to {args.target_host}:{args.target_port}, engine={note}")
    while True:
        session = receiver.receive_one_session()
        if session is None:
            continue
        meta = session.meta

        if meta.coding == "raw-int16":
            assert isinstance(model, RawPassThrough)
            latent_np = depacketize_int16_tensor(session.packets, meta.latent_shape, meta.total_packets, meta.seed)
            latent = torch.from_numpy(latent_np.astype("float32")).unsqueeze(0).to(device)
            with torch.no_grad():
                recoded = model.recode(latent)
            recoded_np = recoded.squeeze(0).detach().cpu().numpy().astype("int16")
            packetized = packetize_int16_tensor(recoded_np, payload_budget(args.mtu), seed=meta.seed)
            out_meta = SessionMeta(
                image_id=meta.image_id,
                filename=meta.filename,
                engine=note,
                total_packets=packetized.total_packets,
                seed=packetized.seed,
                latent_shape=packetized.shape,
                original_size=meta.original_size,
                padded_size=meta.padded_size,
                dtype="int16",
                coding="raw-int16",
                chunk_size=0,
            )
            packets = packetized.packets
        else:
            assert isinstance(model, EntropyWACNN)
            ordered_chunks = depacketize_bytes_chunks(session.packets, meta.total_packets, meta.seed)
            latent = model.decompress_latent_chunks(
                ordered_chunks,
                latent_shape=meta.latent_shape,
                chunk_size=meta.chunk_size,
                device=device,
            )
            with torch.no_grad():
                recoded = model.recode(latent)
                recoded_chunks = model.compress_latent_chunks(recoded, chunk_size=meta.chunk_size)
            ensure_chunk_payload_budget(recoded_chunks, payload_budget(args.mtu))
            packetized = packetize_bytes_chunks(recoded_chunks, seed=meta.seed)
            out_meta = SessionMeta(
                image_id=meta.image_id,
                filename=meta.filename,
                engine=note,
                total_packets=packetized.total_packets,
                seed=packetized.seed,
                latent_shape=tuple(int(x) for x in recoded.shape[1:]),
                original_size=meta.original_size,
                padded_size=meta.padded_size,
                dtype="entropy_bottleneck",
                coding="compressai-eb",
                chunk_size=meta.chunk_size,
            )
            packets = packetized.packets

        meta_packet = build_meta_packet(out_meta)
        for _ in range(3):
            sender.sendto(meta_packet, reliable=True)
            time.sleep(0.02)
        for packet_index, payload in enumerate(packets):
            sender.sendto(build_data_packet(meta.image_id, packet_index, out_meta.total_packets, payload), reliable=False)
            time.sleep(args.send_interval_ms / 1000.0)
        for _ in range(3):
            sender.sendto(build_end_packet(meta.image_id), reliable=True)
            time.sleep(0.02)

        print(
            f"[relay] image_id={meta.image_id} in={len(session.packets)}/{meta.total_packets} pkts "
            f"out={out_meta.total_packets} pkts coding={out_meta.coding} end_received={session.end_received}"
        )


if __name__ == "__main__":
    main()
