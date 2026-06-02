"""
NLA Pipeline
------------
1. Reads sentences from an input TSV (index<tab>sentence)
2. Runs each sentence through Qwen2.5-7B-Instruct-AWQ on GPU 0,
   extracts the last-token hidden state at layer 20
3. Sends that activation to the NLA Activation Verbalizer (AV)
   server on GPU 1 via SGLang, gets back a natural-language "thought"
4. Writes results to an output TSV (index<tab>activation_l2<tab>thought)

Usage:
    python run_nla.py [--input test.txt] [--output result.txt]
"""

import argparse
import os
import time
import httpx
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"  # int4 AWQ — fits in 11GB VRAM
QWEN_LAYER = 20          # layer 20 of 28 — ~2/3 through the model
QWEN_DEVICE = "cuda:0"

AV_MODEL = "kitft/nla-qwen2.5-7b-L20-av"
AV_SGLANG_URL = os.getenv("AV_SGLANG_URL", "http://localhost:30000")
AV_TARGET_NORM = 150.0   # per nla_meta.yaml for this checkpoint
AV_INJECT_CHAR = "㈎"   # injection marker used by the AV prompt template


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run NLA verbalization pipeline.")
    p.add_argument("--input", default="test.txt",
                   help="Input TSV file with columns: index<tab>sentence (default: test.txt)")
    p.add_argument("--output", default="result.txt",
                   help="Output TSV file (default: result.txt)")
    return p.parse_args()


def load_sentences(path: str) -> list[tuple[int, str]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("index"):
                continue
            idx, sentence = line.split("\t", 1)
            rows.append((int(idx), sentence))
    return rows


def extract_last_token_activation(
    model, tokenizer, sentence: str, layer: int, device: str
) -> np.ndarray:
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple of (n_layers+1) tensors [1, seq_len, d_model]
    hidden = out.hidden_states[layer]  # [1, seq_len, d_model]
    last_token = hidden[0, -1, :]     # [d_model]
    return last_token.float().cpu().numpy()


def normalize_activation(vec: np.ndarray, target_norm: float) -> np.ndarray:
    l2 = float(np.linalg.norm(vec))
    if l2 == 0:
        return vec
    return vec * (target_norm / l2)


_av_cache: dict = {}

def _get_av_embed(av_model_path: str):
    if av_model_path not in _av_cache:
        print(f"  [AV] Loading tokenizer + embedding from {av_model_path} ...")
        tok = AutoTokenizer.from_pretrained(av_model_path)
        # Load only the embedding layer on CPU — SGLang already owns GPU 1
        m = AutoModelForCausalLM.from_pretrained(
            av_model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        embed_weight = m.model.embed_tokens.weight.detach()  # [vocab, d_model]
        del m
        _av_cache[av_model_path] = (tok, embed_weight)
    return _av_cache[av_model_path]


def av_verbalize(vec: np.ndarray, sglang_url: str, av_model_path: str) -> str:
    tok, embed_weight = _get_av_embed(av_model_path)

    prompt_template = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Describe the information encoded in this activation: {AV_INJECT_CHAR}<|im_end|>\n"
        f"<|im_start|>assistant\n<explanation>"
    )

    token_ids = tok(prompt_template, return_tensors="pt").input_ids[0]  # [L]
    embeds = embed_weight[token_ids].float().numpy()                     # [L, d_model]

    inject_id = tok.convert_tokens_to_ids(AV_INJECT_CHAR)
    positions = (token_ids == inject_id).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        raise ValueError(f"Injection character '{AV_INJECT_CHAR}' not found in tokenized prompt.")
    embeds[int(positions[0])] = vec

    payload = {
        "input_embeds": embeds.tolist(),
        "sampling_params": {
            "max_new_tokens": 200,
            "temperature": 0.0,
            "stop": ["</explanation>", "<|im_end|>"],
        },
    }

    resp = httpx.post(f"{sglang_url}/generate", json=payload, timeout=60.0)
    resp.raise_for_status()
    text = resp.json().get("text", "")
    if "</explanation>" in text:
        text = text.split("</explanation>")[0]
    return text.strip()


def main():
    args = parse_args()

    sentences = load_sentences(args.input)
    print(f"Loaded {len(sentences)} sentences from {args.input}")

    print(f"\nLoading {QWEN_MODEL} on {QWEN_DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        device_map=QWEN_DEVICE,
    )
    model.eval()
    print("Qwen loaded.")

    print(f"\nWaiting for AV SGLang server at {AV_SGLANG_URL} ...")
    for _ in range(30):
        try:
            r = httpx.get(f"{AV_SGLANG_URL}/health", timeout=3.0)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(5)
    else:
        raise RuntimeError("AV SGLang server did not become ready in time.")
    print("AV server ready.")

    results = []
    for idx, sentence in sentences:
        print(f"\n[{idx}] \"{sentence}\"")

        raw_vec = extract_last_token_activation(model, tokenizer, sentence, QWEN_LAYER, QWEN_DEVICE)
        l2_norm = float(np.linalg.norm(raw_vec))
        print(f"  activation l2={l2_norm:.2f}, dim={raw_vec.shape[0]}")

        norm_vec = normalize_activation(raw_vec, AV_TARGET_NORM)
        thought = av_verbalize(norm_vec, AV_SGLANG_URL, AV_MODEL)
        print(f"  thought: {thought}")

        results.append((idx, l2_norm, thought))

    with open(args.output, "w") as f:
        f.write("index\tactivation_l2\tthought\n")
        for idx, l2, thought in results:
            f.write(f"{idx}\t{l2:.4f}\t{thought}\n")

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
