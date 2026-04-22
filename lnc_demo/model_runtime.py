from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


try:
    from compressai.entropy_models import EntropyBottleneck  # type: ignore
    from compressai.layers import GDN  # type: ignore
    from compressai.models import CompressionModel  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    EntropyBottleneck = None
    GDN = None
    CompressionModel = nn.Module  # type: ignore[misc,assignment]


def conv(in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 2) -> nn.Conv2d:
    padding = kernel_size // 2
    return nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)


def deconv(in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 2) -> nn.ConvTranspose2d:
    padding = kernel_size // 2
    output_padding = stride - 1
    return nn.ConvTranspose2d(
        in_ch,
        out_ch,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )


class EntropyWACNN(CompressionModel):
    """Runtime WACNN that keeps CompressAI entropy coding in the transmission path.

    This follows the uploaded base WACNN structure (not WACNN_Pixel):
    g_a -> entropy bottleneck -> g_a_relay -> entropy bottleneck -> g_s.
    """

    def __init__(self, n_channels: int = 192, m_channels: int = 192, **kwargs: Any) -> None:
        if EntropyBottleneck is None or GDN is None:
            raise RuntimeError(
                "compressai is required for engine=wacnn-demo because this runtime now uses "
                "EntropyBottleneck.compress/decompress in the packet path."
            )
        super().__init__(**kwargs)
        self.N = n_channels
        self.M = m_channels
        self.entropy_bottleneck = EntropyBottleneck(m_channels)
        self.g_a = nn.Sequential(
            conv(3, n_channels, kernel_size=5, stride=2),
            GDN(n_channels),
            conv(n_channels, n_channels, kernel_size=5, stride=2),
            GDN(n_channels),
            conv(n_channels, n_channels, kernel_size=5, stride=1),
            conv(n_channels, n_channels, kernel_size=5, stride=2),
            GDN(n_channels),
            conv(n_channels, m_channels, kernel_size=5, stride=2),
            conv(m_channels, m_channels, kernel_size=5, stride=1),
        )
        self.g_a_relay = nn.Sequential(
            conv(m_channels, m_channels, kernel_size=5, stride=1),
            conv(m_channels, m_channels, kernel_size=5, stride=1),
            conv(m_channels, m_channels, kernel_size=5, stride=1),
        )
        self.g_s = nn.Sequential(
            conv(m_channels, m_channels, kernel_size=5, stride=1),
            deconv(m_channels, n_channels, kernel_size=5, stride=2),
            GDN(n_channels, inverse=True),
            deconv(n_channels, n_channels, kernel_size=5, stride=2),
            GDN(n_channels, inverse=True),
            conv(m_channels, m_channels, kernel_size=5, stride=1),
            deconv(n_channels, n_channels, kernel_size=5, stride=2),
            GDN(n_channels, inverse=True),
            deconv(n_channels, 3, kernel_size=5, stride=2),
        )

    @torch.no_grad()
    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.g_a(x)

    @torch.no_grad()
    def recode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.g_a_relay(latent)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.g_s(latent).clamp_(0.0, 1.0)

    @torch.no_grad()
    def compress_latent_chunks(self, latent: torch.Tensor, chunk_size: int = 16) -> List[bytes]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if latent.ndim != 4 or latent.size(0) != 1:
            raise ValueError(f"Expected latent shape [1, C, H, W], got {tuple(latent.shape)}")
        flattened = latent.reshape(latent.size(0), latent.size(1), -1)
        chunks = torch.split(flattened, chunk_size, dim=2)
        out: List[bytes] = []
        for chunk in chunks:
            strings = self.entropy_bottleneck.compress(chunk)
            if len(strings) != 1:
                raise RuntimeError(f"Only batch size 1 is supported, got {len(strings)} strings")
            out.append(strings[0])
        return out

    @torch.no_grad()
    def decompress_latent_chunks(
        self,
        chunk_payloads: Sequence[Optional[bytes]],
        latent_shape: Tuple[int, int, int],
        chunk_size: int = 16,
        device: str = "cpu",
    ) -> torch.Tensor:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        channels, height, width = (int(latent_shape[0]), int(latent_shape[1]), int(latent_shape[2]))
        total_positions = int(height * width)
        recovered_chunks: List[torch.Tensor] = []
        num_expected = int(math.ceil(total_positions / chunk_size))
        if len(chunk_payloads) != num_expected:
            raise ValueError(f"Expected {num_expected} chunk payloads, got {len(chunk_payloads)}")

        for chunk_index in range(num_expected):
            chunk_len = min(chunk_size, total_positions - chunk_index * chunk_size)
            payload = chunk_payloads[chunk_index]
            if payload is None:
                recovered = torch.zeros((1, channels, chunk_len), dtype=torch.float32, device=device)
            else:
                recovered = self.entropy_bottleneck.decompress([payload], (chunk_len,))
                recovered = recovered.to(device=device, dtype=torch.float32)
            recovered_chunks.append(recovered)

        latent = torch.cat(recovered_chunks, dim=2).reshape(1, channels, height, width)
        return latent


class RawPassThrough(nn.Module):
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.no_grad()
    def recode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.clamp_(0.0, 1.0)


@dataclass
class ImageTensorBundle:
    tensor: torch.Tensor
    original_size: Tuple[int, int]
    padded_size: Tuple[int, int]


def pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, _, h, w = x.shape
    new_h = int(math.ceil(h / multiple) * multiple)
    new_w = int(math.ceil(w / multiple) * multiple)
    if new_h == h and new_w == w:
        return x, (h, w)
    padded = torch.zeros((1, 3, new_h, new_w), dtype=x.dtype, device=x.device)
    padded[:, :, :h, :w] = x
    return padded, (new_h, new_w)


def load_image_as_tensor(path: str | Path, device: str = "cpu") -> ImageTensorBundle:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    padded, padded_size = pad_to_multiple(tensor, multiple=16)
    return ImageTensorBundle(
        tensor=padded,
        original_size=(image.height, image.width),
        padded_size=padded_size,
    )


def tensor_to_pil(tensor: torch.Tensor, original_size: Tuple[int, int]) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    tensor = tensor[:, :, : original_size[0], : original_size[1]]
    arr = (tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def sanitize_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        out[key] = value
    return out


def infer_model_dims(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    n_channels = int(state_dict["g_a.0.weight"].shape[0])
    m_channels = int(state_dict["g_a.6.weight"].shape[0])
    return n_channels, m_channels


def build_engine(
    engine: str,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[nn.Module, str]:
    engine = engine.lower()
    if engine == "raw":
        model: nn.Module = RawPassThrough().to(device)
        model.eval()
        return model, "raw"

    model = EntropyWACNN().to(device)
    note = "wacnn-demo+compressai"

    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device)
        state = ckpt.get("state_dict", ckpt)
        state = sanitize_state_dict(state)
        if "g_a.0.weight" in state and "g_a.6.weight" in state:
            n_channels, m_channels = infer_model_dims(state)
            model = EntropyWACNN(n_channels=n_channels, m_channels=m_channels).to(device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        note = (
            f"wacnn-demo+compressai(checkpoint={ckpt_path.name}, "
            f"missing={len(missing)}, unexpected={len(unexpected)})"
        )

    model.eval()
    if hasattr(model, "update"):
        # CompressAI needs CDF tables before compress/decompress is called.
        model.update(force=True)  # type: ignore[attr-defined]
    return model, note
