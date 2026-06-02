"""
RQ2: NLA Prediction Stability
-----------------------------
Hypothesis: NLA's stability in describing activation(s) reflects the target
model's uncertainty (and, downstream, its propensity to hallucinate).

This pipeline extends run_nla.py. For each example it:

  1. Runs the prompt through Qwen2.5-7B-Instruct-AWQ (GPU 0) and extracts
     the layer-20 residual-stream activation at EVERY token (not just the last).
  2. Computes three NLA stability measures via the AV SGLang server (GPU 1):

       M1  cross-token semantic similarity
           Verbalize each token's activation, embed the thoughts, and measure
           how similar the per-token thoughts are to one another. The NLA paper
           notes that claims recurring across adjacent tokens are more reliable;
           low cross-token similarity => the model's internal "story" is unstable.

       M2  average log-p / perplexity of the AV's output
           For the last-token activation, ask the AV server for token logprobs
           and report mean log-p and perplexity. High perplexity => the
           verbalizer is unsure how to describe the activation.

       M3  single-token sampling stability
           For the last-token activation, sample the AV k times at T=1 and
           measure agreement among the k thoughts (mean pairwise cosine of
           thought embeddings + a semantic-entropy estimate over clusters).
           High disagreement => unstable description => model uncertainty.

  3. Joins each example's metrics with the dataset's hallucination labels
     (halu_test_res, abstantion, correct) and writes one JSONL row per example.

The downstream question (done in a separate offline analysis): do M1/M2/M3
predict halu_test_res?

Usage:
    python run_rq2.py \
        --dataset "Hallulens Dataset/1_qwen7b_inference.jsonl" \
        --eval    "Hallulens Dataset/2_eval_results.json" \
        --output  rq2_metrics.jsonl \
        --limit 50 --k-samples 8 --max-tokens-per-token 64
"""

import argparse
import json
import os
import time
import math
import httpx
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config (mirrors run_nla.py)
# ---------------------------------------------------------------------------
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # full bf16 weights (not AWQ)
QWEN_LAYER = 20
QWEN_DEVICE = "cuda:0"

AV_MODEL = "kitft/nla-qwen2.5-7b-L20-av"
AV_SGLANG_URL = os.getenv("AV_SGLANG_URL", "http://localhost:30000")
AV_TARGET_NORM = 150.0
AV_INJECT_CHAR = "㈎"

# Sentence embedder for semantic similarity (small, CPU-friendly).
EMBED_MODEL = os.getenv("RQ2_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DEVICE = os.getenv("RQ2_EMBED_DEVICE", "cuda:0")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ2 NLA prediction-stability pipeline.")
    p.add_argument("--dataset", default="Hallulens Dataset/1_qwen7b_inference.jsonl",
                   help="JSONL with prompt/answer/generation per example.")
    p.add_argument("--eval", default="Hallulens Dataset/2_eval_results.json",
                   help="Eval JSON with parallel halu_test_res / abstantion lists.")
    p.add_argument("--output", default="rq2_metrics.jsonl",
                   help="Output JSONL: one row of metrics + labels per example.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N examples (0 = all).")
    p.add_argument("--k-samples", type=int, default=8,
                   help="M3: number of AV samples per activation at T=1.")
    p.add_argument("--max-tokens-per-token", type=int, default=64,
                   help="M1: max_new_tokens when verbalizing each token (kept short for speed).")
    p.add_argument("--max-tokens", type=int, default=200,
                   help="M2/M3: max_new_tokens for full last-token thoughts.")
    p.add_argument("--max-seq-tokens", type=int, default=512,
                   help="M1: cap number of token positions verbalized per example.")
    p.add_argument("--temperature", type=float, default=0.5,
                   help="AV sampling temperature for all measures (M3 needs >0 to vary).")
    p.add_argument("--use-chat-template", action="store_true",
                   help="Wrap the prompt with Qwen's chat template before extracting activations.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading + label join
# ---------------------------------------------------------------------------
def load_dataset(dataset_path: str, eval_path: str) -> list[dict]:
    rows = [json.loads(l) for l in open(dataset_path)]
    ev = json.load(open(eval_path))

    halu = ev.get("halu_test_res", [])
    abst = ev.get("abstantion", [])
    raw = ev.get("is_hallucinated_raw_generation", [])

    examples = []
    for i, r in enumerate(rows):
        ex = {
            "idx": i,
            "title": r.get("title"),
            "h_score_cat": r.get("h_score_cat"),
            "prompt": r["prompt"],
            "answer": r.get("answer"),
            "generation": r.get("generation"),
            # labels (defensive about length mismatches)
            "halu_test_res": bool(halu[i]) if i < len(halu) else None,
            "abstantion": bool(abst[i]) if i < len(abst) else None,
            "raw_label": raw[i] if i < len(raw) else None,
        }
        # "true" hallucination = wrong AND not a refusal
        if ex["halu_test_res"] is not None and ex["abstantion"] is not None:
            ex["hallucinated_strict"] = ex["halu_test_res"] and not ex["abstantion"]
        else:
            ex["hallucinated_strict"] = None
        examples.append(ex)
    return examples


# ---------------------------------------------------------------------------
# Qwen activation extraction (all tokens at layer L)
# ---------------------------------------------------------------------------
def build_input_text(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_all_token_activations(
    model, tokenizer, text: str, layer: int, device: str
) -> tuple[np.ndarray, list[str]]:
    """Return [seq_len, d_model] layer-L activations and the decoded tokens."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hidden = out.hidden_states[layer][0]  # [seq_len, d_model]
    toks = tokenizer.convert_ids_to_tokens(inputs.input_ids[0].tolist())
    return hidden.float().cpu().numpy(), toks


def normalize_activation(vec: np.ndarray, target_norm: float) -> np.ndarray:
    l2 = float(np.linalg.norm(vec))
    if l2 == 0:
        return vec
    return vec * (target_norm / l2)


# ---------------------------------------------------------------------------
# AV server interface (text + optional logprobs)
# ---------------------------------------------------------------------------
_av_cache: dict = {}


def _get_av_embed(av_model_path: str):
    if av_model_path not in _av_cache:
        print(f"  [AV] Loading tokenizer + embedding from {av_model_path} ...")
        tok = AutoTokenizer.from_pretrained(av_model_path)
        m = AutoModelForCausalLM.from_pretrained(
            av_model_path, torch_dtype=torch.float32, device_map="cpu",
        )
        embed_weight = m.model.embed_tokens.weight.detach()
        del m
        _av_cache[av_model_path] = (tok, embed_weight)
    return _av_cache[av_model_path]


def _build_av_embeds(vec: np.ndarray, av_model_path: str) -> np.ndarray:
    """Build the [L, d_model] input-embeds for the AV prompt with `vec` injected."""
    tok, embed_weight = _get_av_embed(av_model_path)
    prompt_template = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Describe the information encoded in this activation: {AV_INJECT_CHAR}<|im_end|>\n"
        f"<|im_start|>assistant\n<explanation>"
    )
    token_ids = tok(prompt_template, return_tensors="pt").input_ids[0]
    embeds = embed_weight[token_ids].float().numpy()
    inject_id = tok.convert_tokens_to_ids(AV_INJECT_CHAR)
    positions = (token_ids == inject_id).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        raise ValueError(f"Injection character '{AV_INJECT_CHAR}' not found in tokenized prompt.")
    embeds[int(positions[0])] = vec
    return embeds


def _clean_thought(text: str) -> str:
    if "</explanation>" in text:
        text = text.split("</explanation>")[0]
    return text.strip()


def av_generate(
    vec: np.ndarray,
    av_model_path: str,
    sglang_url: str,
    max_new_tokens: int,
    temperature: float,
    return_logprob: bool = False,
    timeout: float = 120.0,
) -> dict:
    """Call the AV server once. Returns {text, logprobs?}.

    logprobs is the list of generated-token logprobs (floats) when requested.
    """
    embeds = _build_av_embeds(vec, av_model_path)
    sampling = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "stop": ["</explanation>", "<|im_end|>"],
    }
    payload = {"input_embeds": embeds.tolist(), "sampling_params": sampling}
    if return_logprob:
        payload["return_logprob"] = True
        # only need logprobs of generated tokens, not the (injected) prompt
        payload["logprob_start_len"] = len(embeds)

    resp = httpx.post(f"{sglang_url}/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    result = {"text": _clean_thought(data.get("text", ""))}

    if return_logprob:
        meta = data.get("meta_info", {}) or {}
        # SGLang returns output_token_logprobs as [[logprob, token_id, token_text], ...]
        otl = meta.get("output_token_logprobs") or []
        result["logprobs"] = [float(t[0]) for t in otl if t and t[0] is not None]
    return result


# ---------------------------------------------------------------------------
# Embedding-based semantic similarity helpers
# ---------------------------------------------------------------------------
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print(f"  [EMB] Loading {EMBED_MODEL} on {EMBED_DEVICE} ...")
        _embedder = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    return _embedder


def embed_texts(texts: list[str]) -> np.ndarray:
    texts = [t if t else " " for t in texts]
    emb = get_embedder().encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return emb  # [n, d], L2-normalized


def mean_pairwise_cosine(emb: np.ndarray) -> float:
    """Mean off-diagonal cosine similarity. Higher => more agreement/stability."""
    n = emb.shape[0]
    if n < 2:
        return float("nan")
    sims = emb @ emb.T  # already normalized
    iu = np.triu_indices(n, k=1)
    return float(sims[iu].mean())


def semantic_entropy(emb: np.ndarray, threshold: float = 0.7) -> float:
    """Cluster thoughts by cosine threshold; return entropy (nats) over cluster sizes.

    0 => all thoughts mean the same thing (stable). Higher => more distinct
    meanings (unstable). A cheap stand-in for Kuhn et al. semantic entropy.
    """
    n = emb.shape[0]
    if n < 2:
        return 0.0
    sims = emb @ emb.T
    # greedy single-link clustering by similarity threshold
    cluster = [-1] * n
    cid = 0
    for i in range(n):
        if cluster[i] != -1:
            continue
        cluster[i] = cid
        for j in range(i + 1, n):
            if cluster[j] == -1 and sims[i, j] >= threshold:
                cluster[j] = cid
        cid += 1
    counts = np.bincount(cluster)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


# ---------------------------------------------------------------------------
# The three measures
# ---------------------------------------------------------------------------
def measure_cross_token(
    acts: np.ndarray, av_model_path: str, sglang_url: str,
    max_new_tokens: int, max_positions: int, temperature: float,
) -> dict:
    """M1: verbalize each token's activation, embed, measure cross-token agreement."""
    seq_len = acts.shape[0]
    # Evenly subsample token positions if the sequence is long.
    if seq_len > max_positions:
        positions = np.linspace(0, seq_len - 1, max_positions).round().astype(int)
        positions = sorted(set(int(p) for p in positions))
    else:
        positions = list(range(seq_len))

    thoughts = []
    for pos in positions:
        vec = normalize_activation(acts[pos], AV_TARGET_NORM)
        out = av_generate(vec, av_model_path, sglang_url,
                          max_new_tokens=max_new_tokens, temperature=temperature)
        thoughts.append(out["text"])

    emb = embed_texts(thoughts)
    return {
        "n_positions": len(positions),
        "cross_token_mean_cosine": mean_pairwise_cosine(emb),
        "cross_token_semantic_entropy": semantic_entropy(emb),
        "thoughts_per_token": thoughts,
    }


def measure_perplexity(
    last_vec: np.ndarray, av_model_path: str, sglang_url: str,
    max_new_tokens: int, temperature: float,
) -> dict:
    """M2: average log-p / perplexity of the AV's last-token thought."""
    out = av_generate(last_vec, av_model_path, sglang_url,
                      max_new_tokens=max_new_tokens, temperature=temperature,
                      return_logprob=True)
    lps = out.get("logprobs", [])
    if lps:
        mean_logp = float(np.mean(lps))
        perplexity = float(math.exp(-mean_logp))
    else:
        mean_logp = float("nan")
        perplexity = float("nan")
    return {
        "last_token_thought": out["text"],
        "mean_logp": mean_logp,
        "perplexity": perplexity,
        "n_logprob_tokens": len(lps),
    }


def measure_sampling_stability(
    last_vec: np.ndarray, av_model_path: str, sglang_url: str,
    max_new_tokens: int, k: int, temperature: float,
) -> dict:
    """M3: sample the AV k times for the same activation, measure agreement.

    Requires temperature > 0, otherwise every sample is identical and the
    measure degenerates (cosine=1.0, entropy=0).
    """
    samples = []
    for _ in range(k):
        out = av_generate(last_vec, av_model_path, sglang_url,
                          max_new_tokens=max_new_tokens, temperature=temperature)
        samples.append(out["text"])
    emb = embed_texts(samples)
    return {
        "k": k,
        "sampling_mean_cosine": mean_pairwise_cosine(emb),
        "sampling_semantic_entropy": semantic_entropy(emb),
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def wait_for_av(url: str):
    print(f"\nWaiting for AV SGLang server at {url} ...")
    for _ in range(30):
        try:
            r = httpx.get(f"{url}/health", timeout=3.0)
            if r.status_code == 200:
                print("AV server ready.")
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("AV SGLang server did not become ready in time.")


def main():
    args = parse_args()

    examples = load_dataset(args.dataset, args.eval)
    if args.limit > 0:
        examples = examples[: args.limit]
    print(f"Loaded {len(examples)} examples from {args.dataset}")
    print(f"AV temperature={args.temperature}  max_seq_tokens={args.max_seq_tokens}  k={args.k_samples}")
    if args.temperature <= 0.0:
        print("  [WARN] temperature=0 makes M3 (sampling stability) degenerate: "
              "all k samples will be identical (cosine=1.0, entropy=0).")

    print(f"\nLoading {QWEN_MODEL} on {QWEN_DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, torch_dtype=torch.bfloat16, device_map=QWEN_DEVICE,
    )
    model.eval()
    print("Qwen loaded.")

    wait_for_av(AV_SGLANG_URL)

    out_f = open(args.output, "w")
    for ex in examples:
        idx = ex["idx"]
        print(f"\n[{idx}] {ex['prompt'][:80]!r} "
              f"(halu={ex['halu_test_res']} abstain={ex['abstantion']})")

        text = build_input_text(tokenizer, ex["prompt"], args.use_chat_template)
        acts, toks = extract_all_token_activations(
            model, tokenizer, text, QWEN_LAYER, QWEN_DEVICE
        )
        last_vec = normalize_activation(acts[-1], AV_TARGET_NORM)
        print(f"  seq_len={acts.shape[0]} d_model={acts.shape[1]}")

        m1 = measure_cross_token(
            acts, AV_MODEL, AV_SGLANG_URL,
            max_new_tokens=args.max_tokens_per_token,
            max_positions=args.max_seq_tokens,
            temperature=args.temperature,
        )
        m2 = measure_perplexity(
            last_vec, AV_MODEL, AV_SGLANG_URL, max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        m3 = measure_sampling_stability(
            last_vec, AV_MODEL, AV_SGLANG_URL,
            max_new_tokens=args.max_tokens, k=args.k_samples,
            temperature=args.temperature,
        )

        print(f"  M1 cross-token cosine={m1['cross_token_mean_cosine']:.3f} "
              f"entropy={m1['cross_token_semantic_entropy']:.3f}")
        print(f"  M2 perplexity={m2['perplexity']:.2f} "
              f"(mean_logp={m2['mean_logp']:.3f}, n={m2['n_logprob_tokens']})")
        print(f"  M3 sampling cosine={m3['sampling_mean_cosine']:.3f} "
              f"entropy={m3['sampling_semantic_entropy']:.3f}")

        record = {
            # identity + labels
            "idx": idx,
            "title": ex["title"],
            "h_score_cat": ex["h_score_cat"],
            "prompt": ex["prompt"],
            "answer": ex["answer"],
            "generation": ex["generation"],
            "halu_test_res": ex["halu_test_res"],
            "abstantion": ex["abstantion"],
            "hallucinated_strict": ex["hallucinated_strict"],
            "seq_len": int(acts.shape[0]),
            "av_temperature": args.temperature,
            # M1
            "cross_token_mean_cosine": m1["cross_token_mean_cosine"],
            "cross_token_semantic_entropy": m1["cross_token_semantic_entropy"],
            "n_positions": m1["n_positions"],
            # M2
            "mean_logp": m2["mean_logp"],
            "perplexity": m2["perplexity"],
            "n_logprob_tokens": m2["n_logprob_tokens"],
            "last_token_thought": m2["last_token_thought"],
            # M3
            "sampling_mean_cosine": m3["sampling_mean_cosine"],
            "sampling_semantic_entropy": m3["sampling_semantic_entropy"],
            # raw text for inspection
            "thoughts_per_token": m1["thoughts_per_token"],
            "samples": m3["samples"],
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()

    out_f.close()
    print(f"\nMetrics written to {args.output}")


if __name__ == "__main__":
    main()
