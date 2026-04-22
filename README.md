# Three-terminal multi-hop LNC demo (CompressAI entropy-coded version)

This version keeps the same **source / relay / destination** split as before, but the `wacnn-demo` path now uses **CompressAI `EntropyBottleneck.compress/decompress`** in the actual packet path instead of shipping raw `int16` latents.

## What changed in this version

For `--engine wacnn-demo`:

- **Source** runs `g_a`, then entropy-codes latent chunks with `entropy_bottleneck.compress(...)`.
- **Relay** decompresses the received latent chunks, zero-fills missing chunks, runs `g_a_relay`, and entropy-codes the recoded latent again.
- **Destination** decompresses the received recoded latent chunks and runs `g_s`.
- Packet payloads are now **variable-length entropy-coded byte strings**, which is much closer to your own demo snippet and to the paper's *encoder -> entropy model -> packetization -> decoder* pipeline.

This matches the uploaded base `WACNN` structure, where the model contains `EntropyBottleneck`, `g_a`, `g_a_relay`, and `g_s`. The training script also shows the base training path instantiating `WACNN(192, 192)`. fileciteturn9file2 fileciteturn9file3

## Important constraint

This package now assumes:

1. you have **CompressAI installed**, and
2. the checkpoint you pass to `--checkpoint` matches the **base WACNN**


## Installation

```bash
pip install torch pillow numpy compressai
```

## Recommended first check

```bash
cd lnc_three_terminal_demo
PYTHONPATH=. python check_checkpoint.py --checkpoint /path/to/checkpoint_best.pth
```

## Run on three terminals

### Terminal 1: destination

```bash
cd lnc_three_terminal_demo
PYTHONPATH=. python destination.py \
  --bind-port 9102 \
  --engine wacnn-demo \
  --checkpoint /path/to/checkpoint_best.pth \
  --entropy-chunk 16 \
  --output-dir outputs
```

### Terminal 2: relay

```bash
cd lnc_three_terminal_demo
PYTHONPATH=. python relay.py \
  --bind-port 9101 \
  --target-host 127.0.0.1 \
  --target-port 9102 \
  --engine wacnn-demo \
  --checkpoint /path/to/checkpoint_best.pth \
  --entropy-chunk 16 \
  --drop-prob 0.1
```

### Terminal 3: source

```bash
cd lnc_three_terminal_demo
PYTHONPATH=. python source.py \
  --target-host 127.0.0.1 \
  --target-port 9101 \
  --image /path/to/test.png \
  --engine wacnn-demo \
  --checkpoint /path/to/checkpoint_best.pth \
  --entropy-chunk 16 \
  --drop-prob 0.1
```

## Chunk-size tuning

Each entropy-coded chunk is sent as **one UDP packet** in this demo. If you see an error saying one chunk exceeds the UDP payload budget, lower `--entropy-chunk`, for example:

```bash
--entropy-chunk 8
```

This is the main knob controlling the trade-off between:

- fewer/larger packets, and
- safer per-packet payload size.

