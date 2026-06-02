# NLA Pipeline

Runs sentences through **Qwen2.5-7B-Instruct-AWQ** (GPU 0), extracts last-token hidden states at layer 20, then feeds them to the **NLA Activation Verbalizer** (GPU 1) to get natural-language "thoughts". Results are written to a TSV.

---

## Files

| File | Purpose |
|------|---------|
| `test.txt` | Input sentences (TSV: `index\tsentence`) |
| `run_nla.py` | Basic pipeline script (one thought per sentence) |
| `run_rq2.py` | **RQ2** stability pipeline (3 measures + dataset-label join) |
| `result.txt` | `run_nla.py` output (`index\tactivation_l2\tthought`) |
| `rq2_metrics.jsonl` | `run_rq2.py` output (one metrics+labels row per example) |
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

---

## RQ2: NLA Prediction Stability (`run_rq2.py`)

**Hypothesis:** NLA's stability in describing activations reflects the target
model's uncertainty — and therefore its propensity to hallucinate.

For each example in the HalluLens dataset, `run_rq2.py` extracts Qwen's layer-20
activations and computes three stability measures via the AV server, then joins
each row with the dataset's hallucination labels so the signals can be
correlated against ground truth offline.

| Measure | What it computes | Intuition |
|---------|------------------|-----------|
| **M1 cross-token similarity** | Verbalize *every* token's activation, embed the thoughts, report mean pairwise cosine + a semantic-entropy estimate | The NLA paper notes claims recurring across adjacent tokens are more reliable; low cross-token agreement ⇒ unstable internal "story" |
| **M2 perplexity** | Ask the AV for token logprobs on the last-token thought; report mean log-p and `exp(−mean log-p)` | High perplexity ⇒ the verbalizer is unsure how to describe the activation |
| **M3 sampling stability** | Sample the AV `k` times at `T=1` for the same activation; report mean pairwise cosine + semantic entropy over the `k` thoughts | High disagreement across samples ⇒ unstable description ⇒ model uncertainty |

### Run

The AV SGLang server must be started **with logprobs enabled** (M2 needs them):

```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
    --model-path kitft/nla-qwen2.5-7b-L20-av \
    --port 30000 --disable-radix-cache --dtype bfloat16
```

Also install the sentence embedder used for M1/M3 similarity:

```bash
pip install sentence-transformers
```

Then:

```bash
# smoke test on the first 10 examples
python run_rq2.py --limit 10 --k-samples 4

# full run
python run_rq2.py \
    --dataset "Hallulens Dataset/1_qwen7b_inference.jsonl" \
    --eval    "Hallulens Dataset/2_eval_results.json" \
    --output  rq2_metrics.jsonl \
    --k-samples 8 --max-tokens-per-token 64 --max-seq-tokens 64
```

Add `--use-chat-template` to wrap each prompt with Qwen's chat template before
extracting activations (matches how the answers in `1_qwen7b_inference.jsonl`
were generated).

### Output (`rq2_metrics.jsonl`)

One JSON object per example, e.g.:

```json
{
  "idx": 1, "prompt": "What was Real Chemistry formerly known as?",
  "answer": "W2O Group", "halu_test_res": true, "abstantion": false,
  "hallucinated_strict": true,
  "cross_token_mean_cosine": 0.41, "cross_token_semantic_entropy": 1.79,
  "mean_logp": -1.83, "perplexity": 6.23,
  "sampling_mean_cosine": 0.52, "sampling_semantic_entropy": 1.10,
  "last_token_thought": "...", "thoughts_per_token": ["..."], "samples": ["..."]
}
```

`hallucinated_strict = halu_test_res AND NOT abstantion` — refusals are excluded
from the hallucination set (in the raw eval, an abstention is sometimes counted
as a hallucination). Use this field as the prediction target. The expected
finding: hallucinated examples show **lower** cross-token/sampling cosine and
**higher** entropy/perplexity than correct ones.
