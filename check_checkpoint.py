from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lnc_demo.model_runtime import build_engine, sanitize_state_dict, infer_model_dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inspect an LNC/WACNN checkpoint and test-load it into the demo runtime.')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    ckpt = torch.load(str(ckpt_path), map_location=args.device)
    state = ckpt.get('state_dict', ckpt)
    state = sanitize_state_dict(state)
    print(f'[check] checkpoint={ckpt_path}')
    print(f'[check] top-level type={type(ckpt).__name__}')
    print(f'[check] state tensors={len(state)}')
    if 'g_a.0.weight' in state and 'g_a.6.weight' in state:
        n_channels, m_channels = infer_model_dims(state)
        print(f'[check] inferred dims: N={n_channels}, M={m_channels}')
    else:
        print('[check] could not infer dims from g_a weights')

    model, note = build_engine('wacnn-demo', checkpoint=str(ckpt_path), device=args.device)
    print(f'[check] load note: {note}')
    first_keys = list(state.keys())[:20]
    print('[check] first keys:')
    for key in first_keys:
        print('  ', key)


if __name__ == '__main__':
    main()
