# NLA Pipeline

Runs sentences through **Qwen2.5-7B-Instruct-AWQ** (GPU 0), extracts last-token hidden states at layer 20, then feeds them to the **NLA Activation Verbalizer** (GPU 1) to get natural-language "thoughts". Results are written to a TSV.

---

## Files

| File | Purpose |
|------|---------|
| `test.txt` | Input sentences (TSV: `index\tsentence`) |
| `run_nla.py` | Pipeline script |
| `result.txt` | Output (created after a run: `index\tactivation_l2\tthought`) |
| `Dockerfile` | Container for the pipeline runner |
| `docker-compose.yml` | Orchestrates AV server (GPU 1) + pipeline (GPU 0) |

---

## Why AWQ?

The Qwen2.5-7B model in bfloat16 needs ~14 GB VRAM but the RTX 2080 Ti has 11 GB. The official `Qwen/Qwen2.5-7B-Instruct-AWQ` (int4) cuts that to ~4.5 GB, leaving headroom for the hidden-state extraction. The AV model runs separately on GPU 1 via SGLang.

---

## Running with Docker (recommended)

### Prerequisites

- Docker + `docker compose` v2
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on the host

### Basic run (defaults: `test.txt` → `result.txt`)

```bash
docker compose up --build
```

`result.txt` appears in the project directory when the pipeline finishes.

### Custom input / output

```bash
INPUT_FILE=data/my_sentences.txt OUTPUT_FILE=data/my_results.txt docker compose up --build
```

Paths are relative to the project directory (mounted at `/app/data` inside the container).

---

## Running directly (no Docker)

### 1. Install dependencies

```bash
pip install torch transformers accelerate autoawq httpx numpy "sglang[all]>=0.5.6"
```

### 2. Start the AV SGLang server on GPU 1

```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
    --model-path kitft/nla-qwen2.5-7b-L20-av \
    --port 30000 \
    --disable-radix-cache \
    --dtype bfloat16
```

Wait for `Server is ready` in the logs.

### 3. Run the pipeline

```bash
# defaults
python run_nla.py

# custom files
python run_nla.py --input my_sentences.txt --output my_results.txt
```

The `AV_SGLANG_URL` env var controls the server address (default: `http://localhost:30000`).

---

## Input format

Plain TSV, one sentence per line, header optional:

```
index	sentence
0	The quick brown fox jumps over the lazy dog.
1	Artificial intelligence is reshaping how humans interact with machines.
```

---

## Output format

```
index	activation_l2	thought
0	142.3801	The token is processing a sentence about a fast animal performing an action...
1	138.9204	The representation encodes concepts related to technology and human interaction...
```

---

## GPU assignment

| Service | GPU | Model |
|---------|-----|-------|
| Pipeline runner | GPU 0 | Qwen2.5-7B-Instruct-AWQ (int4, ~4.5 GB) |
| AV SGLang server | GPU 1 | kitft/nla-qwen2.5-7b-L20-av (bfloat16) |
