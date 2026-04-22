from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lnc_demo.common import add_common_model_args, add_common_network_args, create_udp_socket, choose_device, ensure_dir, set_deterministic_seed
from lnc_demo.model_runtime import EntropyWACNN, RawPassThrough, build_engine, tensor_to_pil
from lnc_demo.packetization import depacketize_bytes_chunks, depacketize_int16_tensor
from lnc_demo.protocol import SessionReceiver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LNC destination node: latent packets -> image reconstruction")
    add_common_network_args(parser)
    add_common_model_args(parser)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--recv-timeout-s", type=float, default=30.0)
    parser.add_argument("--idle-timeout-s", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic_seed(args.seed)
    device = choose_device(args.device)
    model, note = build_engine(args.engine, checkpoint=args.checkpoint, device=device)
    out_dir = ensure_dir(args.output_dir)
    sock = create_udp_socket(args.bind_host, args.bind_port)
    receiver = SessionReceiver(sock, timeout_s=args.recv_timeout_s, idle_timeout_s=args.idle_timeout_s)

    print(f"[destination] listening on {args.bind_host}:{args.bind_port}, saving to {out_dir}, engine={note}")
    while True:
        session = receiver.receive_one_session()
        if session is None:
            continue
        meta = session.meta

        if meta.coding == "raw-int16":
            assert isinstance(model, RawPassThrough)
            latent_np = depacketize_int16_tensor(session.packets, meta.latent_shape, meta.total_packets, meta.seed)
            latent = torch.from_numpy(latent_np.astype("float32")).unsqueeze(0).to(device)
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
            recon = model.decode(latent)
        image = tensor_to_pil(recon, meta.original_size)
        out_path = out_dir / f"recv_{meta.image_id}_{Path(meta.filename).stem}.png"
        image.save(out_path)
        print(
            f"[destination] image_id={meta.image_id} received={len(session.packets)}/{meta.total_packets} "
            f"saved={out_path} coding={meta.coding} engine={note}"
        )


if __name__ == "__main__":
    main()
