# RQ2 Design Notes — what's spec'd vs. what we engineered

> Working memory for the RQ2 sub-project (Group 17, Hallu-NLA). Captures, for
> each of the three "Implementations" in the proposal slides, exactly what the
> slide pins down versus what we chose ourselves — plus decisions made and open
> questions. Last updated 2026-06-03.

**Hypothesis (from slides / `info.md`):** NLA's stability in describing
activation(s) reflects the target model's uncertainty — and therefore its
propensity to hallucinate.

**Pipeline:** `run_rq2.py` extracts Qwen2.5-7B-Instruct layer-20 activations,
verbalizes them through the vendored official NLA client (`nla_inference.py`,
`NLAClientLP` subclass adds logprobs), computes three measures, and joins each
row to the balanced HalluLens inline labels (`is_hallucinated` / `is_abstaining`
→ `hallucinated_strict = wrong AND not refusal`). Dataset:
`Balanced Hallulens Dataset/balanced_dataset.jsonl` (400 rows, 200/200 balanced,
no abstentions). Output: `rq2_metrics.jsonl` (+ `rq2_metrics.last_act.npy` =
last-token raw activations, row i ↔ example idx i).

---

## The three measures: slide spec vs. our implementation

### M1 — Implementation 1: "semantic similarity of NLA predictions across tokens"
`measure_cross_token`

- **Slide says:** sentence with arrows from *multiple token positions*; caption
  "Check if NLA predictions are semantically similar across tokens." Concept only.
- **We implemented:** verbalize each token position's layer-20 activation → one
  thought per token; embed; report **mean pairwise cosine** + **semantic entropy**.
- **Ours (not in slide):** the embedder, mean-cosine as the aggregation, the
  entropy proxy, and `linspace` subsampling when `seq_len > max_seq_tokens` (512).
- **Faithful?** Yes to the concept; metric implementation is our choice.

### M2 — Implementation 2: "average log p (i.e., perplexity) of NLA's output"
`measure_perplexity`

- **Slide says:** arrow into NLA + a **per-token table** of `log p`; caption
  "average log p (i.e., perplexity) of NLA's output." **Most-specified** of the
  three — it names the metric.
- **We implemented:** verbalize the **last-token** activation, get token logprobs
  from SGLang, report `mean(logprobs)` and `exp(-mean)` (= perplexity).
- **Ours (not in slide):** last-token-only (the slide's table is per-token);
  reading "log p of NLA's output" as the AV's generated-token logprobs.
- **Faithful?** Yes — metric matches the slide. Lowest-risk measure.

### M3 — Implementation 3: "stability of NLA predictions for a single token"
`measure_sampling_stability`

- **Slide says:** one token's activation → **four different candidate
  descriptions** → caption "Stability of NLA predictions for a single token."
  Concept sketch only — **no metric, no method**.
- **We implemented:** take one activation (last token), sample the AV **k=8**
  times (variation from temperature), embed, report **mean pairwise cosine** +
  **semantic entropy** (cluster threshold 0.8).
- **Ours (not in slide):** essentially everything — k=8, embedder, cosine,
  clustering entropy, the threshold.
- **Faithful?** Yes to the concept; **most "designed by us"** of the three. Be
  ready to justify the stability metric, or align it with RQ1's named options
  (BERTScore / Semantic Entropy / Cosine) for cross-paper consistency.

### Summary table

| | Concept from slide | Metric named in slide | Our engineering |
|---|---|---|---|
| **M1** | ✅ similarity across tokens | ❌ ("semantic similarity") | embedder, mean-cosine, entropy, linspace subsample |
| **M2** | ✅ | ✅ **avg log p = perplexity** | last-token-only vs per-token table |
| **M3** | ✅ single-token stability | ❌ (picture only) | k=8, embedder, cosine, entropy, 0.8 threshold |

> Note: the **only** place the proposal names metrics is the **RQ1** slide:
> "BERTScore, Semantic Entropy, or Cosine Similarity." Nothing on the RQ2 slides
> names an embedder, an aggregation, or a threshold. So our metric choices are
> faithful *engineering* of open methods, not deviations from the spec.

---

## Decisions made (and why)

- **Full bf16 Qwen** (`Qwen/Qwen2.5-7B-Instruct`), not AWQ — faithful activations.
- **Vendored official NLA client** (`kitft/nla-inference`, Apache-2.0) instead of
  hand-rolled injection. The client reads `nla_meta.yaml` and handles the trained
  prompt template, neighbor-verified injection position, injection scale, and
  embed scale automatically. Fixed our two earlier bugs (wrong prompt template;
  marker-id crash). We subclass only to add logprobs for M2.
- **Temperature = 1.0** for all three measures. The AV is trained to sample at
  T=1; perplexity (M2) is temperature-dependent, so T=0.5 made it incomparable to
  the paper and biased M1/M3 agreement high. T=1 still keeps M3 non-degenerate
  (needs >0). (Earlier we used 0.5; changed after the audit.)
- **max_new_tokens = 512** for all measures, so the AV emits the closing
  `</explanation>` tag instead of truncating (fixed the "no <explanation> tags"
  warning) + `_strip_explanation_tags()` defense so leftover tags don't pollute
  embeddings.
- **Embedder = all-mpnet-base-v2** (was all-MiniLM-L6-v2). MiniLM scored
  stylistically-similar but semantically-different thoughts as similar (e.g. four
  thoughts about different songs at ~0.75 cosine), biasing M1/M3. mpnet separates
  meaning better. Override via `RQ2_EMBED_MODEL`.
- **Cluster threshold = 0.8** (`SEMANTIC_CLUSTER_THRESHOLD`, was hardcoded 0.7).
  0.7 collapsed everything into one cluster with MiniLM (entropy ~0). 0.8 suits
  mpnet's wider spread. Still hand-picked, not calibrated. Override via
  `RQ2_CLUSTER_THRESHOLD`.

## Audit result (general-purpose agent, 2026-06-02)

No blocking correctness bugs. Verified correct: layer index (`hidden_states[20]`
matches upstream, no off-by-one); single-pass normalization (we pass raw, client
rescales to 150 once); M2 `logprob_start_len` scores only generated tokens; M3
resamples the *same* single activation.

---

## Open questions (decide after a real T=1 + mpnet run)

1. **M3 entropy threshold (0.8) is uncalibrated.** Right fix is to derive it from
   the observed pairwise-cosine distribution, not a magic constant. Treat
   `sampling_mean_cosine` as the PRIMARY M3 signal; entropy is secondary.
2. **Embedder validity.** Even mpnet may track wording over meaning to some
   degree. If cosines stay compressed, consider an NLI-entailment agreement
   metric (closer to Kuhn et al. semantic entropy) for M1/M3.
3. **M1 linspace subsampling** only triggers at `seq_len > 512`; HalluLens prompts
   are 8-25 tokens, so it ~never fires. Left as-is. Revisit only for long inputs.
4. **Confabulation** (NLA describing unrelated content) is independent of all the
   above — likely genuine NLA noise on short prompts (paper: first ~10 token
   positions are unreliable; our sequences are short, so a large fraction is in
   that noisy region). Not a code issue.

## Next step

Re-run the smoke test at T=1 + mpnet and inspect: (a) is the `<explanation>` tag
leak gone, (b) do M3 cosines spread out more, (c) does entropy now vary at 0.8,
(d) does confabulation persist. Then decide on threshold calibration / embedder.

```bash
python run_rq2.py --limit 10 --k-samples 4 --max-seq-tokens 64
```
Watch the `[NLAClient]` startup banner (expect `inj_scale=150.0 embed_scale=1.00
inj_char='㈎'(id=149705)`) and confirm the first thought reads like 2-3
descriptive snippets, not gibberish/CJK.
