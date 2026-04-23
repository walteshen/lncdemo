# Three-terminal multi-hop LNC demo (CompressAI entropy-coded version)

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
cd lncdemo
PYTHONPATH=. python check_checkpoint.py --checkpoint /path/to/checkpoint_best.pth
```

## Run on three terminals

### Terminal 1: destination

```bash
cd lncdemo
PYTHONPATH=. python destination.py \
  --bind-port 9102 \
  --engine wacnn-demo \
  --checkpoint /path/to/checkpoint_best.pth \
  --entropy-chunk 16 \
  --output-dir outputs
```

### Terminal 2: relay

```bash
cd lncdemo
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
cd lncdemo
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

## Checkpoint
lambda=0.013:<https://drive.google.com/file/d/1Dio8SE9JJEI51EA2QndwzUuylfm0M9ua/view?usp=drive_link>
