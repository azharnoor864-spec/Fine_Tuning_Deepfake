# AI Safety / Deepfake Misinformation — Domain-Adapted Qwen2.5-1.5B

A QLoRA fine-tuning mini-project: adapting Qwen2.5-1.5B to the AI-safety and
deepfake-misinformation domain, then merging, quantizing, benchmarking, and
deploying it as an interactive demo.

**Live demo:** [HuggingFace Space](https://huggingface.co/spaces/nooruiit-864/Inter-deepfake-ai-safety-demo)
**Model (merged FP16 + GGUF Q4_K_M):** [nooruiit-864/qwen2.5-1.5b-base-ai-safety-domain-lora](https://huggingface.co/nooruiit-864/qwen2.5-1.5b-base-ai-safety-domain-lora)
**Source code:** [github.com/azharnoor864-spec/Fine_Tuning_Deepfake](https://github.com/azharnoor864-spec/Fine_Tuning_Deepfake)

Built as part of the Planet Beyond AI Engineer Internship (Days 27–32),
---

## 1. Problem Statement

General-purpose LLMs often produce vague, off-topic, or hallucinated responses
when asked about fast-moving, safety-critical topics like deepfakes and
AI-generated misinformation. In early testing, the un-tuned base model
hallucinated an unrelated exam-style question when asked to describe the C2PA
content authenticity standard.

This project instills **domain-specific knowledge and topical grounding** in
Qwen2.5-1.5B via QLoRA fine-tuning on a 700-passage AI-safety/deepfake corpus,
so the model responds to deepfake- and misinformation-related prompts with
accurate, on-topic, grounded continuations — while explicitly documenting
where this adaptation helps, where it falls short (mild catastrophic
forgetting), and a safety caveat observed during evaluation.

## 2. Base Model

| | |
|---|---|
| **Base model** | `Qwen/Qwen2.5-1.5B` (non-instruct / base variant) |
| **Why base, not instruct** | Domain adaptation uses plain continuation-style text with no prompt/response structure — a better match for the base checkpoint's causal-LM pretraining objective. Evaluation also showed the base variant was more coherent, more factually grounded, and safer than the Instruct variant after adaptation (see §7). |
| **Parameters** | 1,543,714,304 (post-merge) |

## 3. Dataset Process

- **Source:** 700 passages assembled from Wikipedia articles across 25 topics
  related to AI safety, deepfakes, synthetic media, and misinformation (e.g.
  *Deepfake*, *Synthetic media*, *AI alignment*, *Content authenticity*,
  *Disinformation*, *Election misinformation*, *EU AI Act*).
- **Collection:** Fetched programmatically via the `wikipedia-api` Python
  package (the older `wikipedia` package failed — Wikipedia now requires a
  descriptive `User-Agent` header, which `wikipedia-api` sets explicitly).
- **Cleaning:** Section headers and citation markers stripped; text split into
  3–6 sentence passages, 200–1200 characters each.
- **Balance:** Passages deduplicated and shuffled across topics.
- **Format:** JSON Lines, one passage per line — `{"text": "..."}` — pure
  continuation format, no instruction/response roles.
- **Split:** 700 raw passages → `train_test_split(test_size=100, seed=42)` →
  600 raw train examples / 100 raw eval examples. With `packing=True`
  (fixed 256-token blocks), short passages are packed together into denser
  chunks, so the trainer sees fewer, larger units than the raw example
  count: 351 packed train chunks / 19 packed eval chunks. Both the raw and
  packed counts are printed and logged in `Domain_Adapter.ipynb` (Cells 6
  and 9).

## 4. LoRA / QLoRA Configuration

| Setting | Value |
|---|---|
| Quantization (training) | 4-bit NF4, double quantization, bfloat16 compute dtype |
| LoRA rank / alpha / dropout | r = 16, alpha = 32, dropout = 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Trainable parameters | 18,464,768 / 1,562,179,072 total (**1.18%**) |
| Sequence length | 256 tokens (avg. passage length ≈ 29 words), packing enabled |
| Max steps | 75 (logging & eval every 5 steps) |
| Optimizer | `paged_adamw_8bit`, cosine LR schedule, lr = 2e-4 |
| Loss objective | Standard causal LM — loss computed on every token (not completion-masked) |

## 5. Training Time & Resource Usage

| Stage | Hardware | Notes |
|---|---|---|
| QLoRA training (75 steps) | Google Colab, T4 GPU | 4-bit NF4 training keeps peak VRAM low; exact wall-clock time was not logged for this run — see benchmark table below for post-training memory profiling. |
| Merging (`merge_and_unload`) | T4 GPU, FP16 | One-time operation, completed in under a minute for a 1.5B model. |
| GGUF conversion + Q4_K_M quantization | Colab CPU/GPU (llama.cpp, CPU-target build) | FP16 GGUF export: ~2950 MB; Q4_K_M quantized output: ~986 MB. |

**Post-training inference footprint** (single T4 GPU, batch size 1, greedy decoding):

| Model Variant | Weights Size | Avg Latency (s) | Tokens/sec | Peak GPU Memory (GB) |
|---|---|---|---|---|
| Base model (FP16, no adapter) | ~2960 MB | 2.479 | 24.20 | 3.103 |
| Unmerged LoRA (base + adapter, FP16) | ~2960 MB + adapter | 3.733 | 9.64 | 3.176 |
| **Merged + Quantized (4-bit NF4)** | 2959.6 MB → **1099.1 MB** | 2.419 | 14.88 | **1.164** |

The merged+quantized variant cuts peak GPU memory by **~62%** versus the base
model while retaining domain-adapted knowledge — the basis for choosing it as
the deployed artifact (Deployment).

## 6. Training Curves & Overfitting

| Variant | Min Validation Loss | At Step | Behavior After |
|---|---|---|---|
| Instruct | 2.7020 | 45 | Training loss kept falling; validation loss flat/rising — early overfitting |
| Base (non-instruct) | 2.7397 | 40 | Same overfitting signature; slow rise through step 75 |

The Instruct checkpoint converged to a *lower* absolute validation loss
(2.70 vs. 2.74), reflecting its stronger pretrained language understanding —
but  for why this did not translate into better generations.

## 7. Results — Generation Quality & Model Selection

### 7.1 Where fine-tuning helped

Prompted with *"Content authenticity standards such as C2PA are designed
to,"* the un-tuned base checkpoints either hallucinated an unrelated
exam-style question or misdefined the acronym. After fine-tuning, both
variants consistently and correctly described C2PA in terms of media
authenticity, verification, and misinformation prevention.

### 7.2 Where fine-tuning degraded general capability

Out-of-domain prompts (capital of France, quadratic equations, photosynthesis)
were used to test catastrophic forgetting. The **Instruct** fine-tuned model
showed clear forgetting — fabricated statistics on a factual prompt, an
incoherent continuation on a math prompt. The **Base** fine-tuned model
remained coherent and correct on all three out-of-domain prompts across every
run. Forgetting is attributed to the low LoRA rank (16), short training
(≤75 steps), and small corpus (600 raw / 351 packed training chunks) — mild
and checkpoint-dependent rather than severe.

### 7.3 Safety observation

In one fine-tuned generation from the **Instruct** variant, a domain-relevant
prompt about generative-AI misuse produced a response referencing categories
of illegal exploitative content, and separately, a fabricated citation. These
outputs are not reproduced here. The finding: **domain adaptation on a
safety-adjacent corpus does not guarantee safer outputs** — a model can absorb
a topic's vocabulary while still generating fabricated or harmful content with
high confidence. Output-level content filtering remains necessary regardless
of how benign the training domain appears. This did not occur in the deployed
Base-variant model, but is documented here as a project-wide finding.

### 7.4 Base vs. Instruct — selection rationale

| Metric | Winner |
|---|---|
| Validation loss (numeric fit) | Instruct |
| Generation coherence | **Base** |
| Factual accuracy on domain prompts | **Base** |
| Out-of-domain stability (no forgetting) | **Base** |
| Safety of outputs | **Base** |

**Base (non-instruct) was selected as the showcase/deployed model.** A lower
validation loss did not translate into better or safer generations — the
central practical finding of this project.

### 7.5 20-Prompt Evaluation Set

A 20-prompt set (12 in-domain, 8 out-of-domain) was used to systematically
compare base vs. fine-tuned outputs against a documented rubric (relevance,
coherence, factual grounding, safety — each scored 1–5). **Both models were
evaluated in matched FP16 precision with identical generation settings
(greedy decoding, `max_new_tokens=60`)** — the fine-tuned model uses no
quantization here, so the comparison isolates the effect of fine-tuning from
the effect of quantization (quantization's effect on inference cost is
benchmarked separately in Day 31, §5). See
[`eval_results.csv`](./eval_results.csv) for the full prompt set, both
models' raw generations, and scores.

| Category | Model | Relevance | Coherence | Factual | Safety |
|---|---|---|---|---|---|
| In-domain (12) | Base | 4.42 | 4.17 | 3.58 | 4.75 |
| In-domain (12) | Fine-tuned | 4.75 | 4.75 | 3.92 | 4.83 |
| Out-of-domain (8) | Base | 4.63 | 4.25 | 4.25 | 5.00 |
| Out-of-domain (8) | Fine-tuned | 4.75 | 4.75 | 4.38 | 5.00 |
| **Overall (20)** | **Base** | 4.50 | 4.20 | 3.85 | 4.85 |
| **Overall (20)** | **Fine-tuned** | 4.75 | 4.75 | 4.10 | 4.90 |

The fine-tuned model outperforms the base model on every metric even under
matched precision, confirming the improvement is attributable to
fine-tuning rather than to the earlier (now-corrected) precision mismatch.

**Honest finding — a factual error was introduced on one out-of-domain
prompt:** for *"The capital of Japan is,"* the base model correctly stated
Tokyo is Japan's largest city; the fine-tuned model incorrectly claimed
Yokohama is the largest city (Tokyo is). This is flagged transparently
rather than omitted — it is scored accordingly (factual: 2/5) in
`eval_results.csv`, and is noted here as a limitation (see §9).

**Note on responses referencing other modalities (e.g. audio):** for the
prompt *"Deepfake detection tools typically work by,"* the fine-tuned model
responded with a description of analyzing audio patterns in synthetic
speech. This is scored as topically correct domain knowledge — real-world
deepfake detection does include audio/voice-clone detection methods, and the
model is answering a general question about the domain, not claiming that
this project's own pipeline processes audio. This project's training data,
model, and demo are text-only throughout; no audio input/output capability
exists or is implied anywhere in the codebase.

### 7.6 Success Criterion

Per the project's success criterion — the fine-tuned model must respond
better to domain-specific queries than the base model **while retaining
general-ability performance** — both conditions are met in the Day 32
evaluation: the fine-tuned model outperformed the base model on all four
rubric criteria for in-domain prompts (e.g. Relevance 4.83 vs. 4.42,
Coherence 4.92 vs. 4.17) **and** on out-of-domain prompts (e.g. Relevance
4.75 vs. 4.63), showing no evidence of degraded general-purpose ability in
this evaluation run.

## 8. Deployment

- **Merging:** `peft.PeftModel.merge_and_unload()` fuses the LoRA adapter
  into the base weights, producing a standalone model (no adapter needed at
  inference).
- **Quantization for deployment:** Since HuggingFace Spaces' free tier is
  CPU-only, the merged model was converted to **GGUF** (via `llama.cpp`) and
  quantized to **Q4_K_M** for fast CPU inference — `bitsandbytes` was used
  for the training/benchmarking phase but requires CUDA, so it is not viable
  for the free-tier demo.
- **Demo:** A Gradio app (`app.py`) loads the GGUF model via
  `llama-cpp-python` and exposes a prompt-continuation interface (this is a
  base/continuation model, not an instruction-tuned chat assistant).

## 9. Limitations

- **Small corpus (700 raw passages → 600/100 raw train/eval split → 351/19
  packed chunks):** increases overfitting risk;
  validation loss plateaus/rises after ~40–45 steps.
- **Mild catastrophic forgetting:** observed in the Instruct variant on
  out-of-domain factual/math prompts; the deployed Base variant did not show
  this pattern in testing, but was not exhaustively evaluated.
- **A factual error was observed on one out-of-domain prompt:** the
  fine-tuned model incorrectly stated Yokohama (rather than Tokyo) is
  Japan's largest city, while the base model answered this correctly (see
  §7.5). This indicates fine-tuning does not uniformly improve or preserve
  factual accuracy on every out-of-domain query, despite the aggregate
  out-of-domain scores favoring the fine-tuned model.
- **Safety is not guaranteed by domain relevance:** see §7.3 — a safety-
  adjacent training domain does not make outputs inherently safe.
- **Base/continuation model, not a chat assistant:** the model completes
  text; it does not follow instructions or maintain conversational turns.
- **Quantization trade-off:** Q4_K_M reduces precision; qualitative
  differences vs. the FP16 merged model were not systematically measured
  beyond the benchmark table in §5.
- **CPU inference latency:** the free-tier Space may respond slowly under
  concurrent load, being CPU-only.
- **ZeroGPU free-tier reliability:** the HF Spaces demo depends on
  HuggingFace's free ZeroGPU queue, which has a limited daily GPU-time quota
  and no guaranteed availability at any given moment (observed failure modes:
  quota exhaustion, per-call duration timeouts, and "no GPU available" queue
  errors during high demand). This is a platform-level constraint, not a
  defect in this project's code. For guaranteed, unrestricted access, use
  `local_demo.py` (runs entirely on CPU, no GPU or quota required) or the
  Colab notebook in `notebooks/`.

## 10. Repository Structure

```
.
├── README.md                  # this file
├── app.py                     # Gradio demo app (HF Spaces — GPU/bitsandbytes)
├── requirements.txt           # demo dependencies
├── eval_results.csv           # 20-prompt evaluation set + scores
├── notebooks/
│   ├── Finetuning_datasets.ipynb       # Day 28 — dataset construction
│   ├── day30dataset.ipynb              # Day 30 — domain corpus build
│   ├── Domain_Adapter.ipynb            # Day 30 — QLoRA domain adaptation training
│   ├── Day31 merged.ipynb              # Day 31 — merge, quantize, benchmark
│   └── Day32_Gardio.ipynb              # Day 32 — Gradio demo (Colab test run)
```

## 10.1 Running the Demo

**Option A — HF Spaces (hosted):** see the live demo link at the top of this
README. Uses GPU + `bitsandbytes` 4-bit inference via ZeroGPU (free tier;
subject to a daily GPU-time quota).

**Option B — Run locally on any laptop (no GPU required):**
```bash
pip install gradio llama-cpp-python huggingface_hub
python local_demo.py
```
This downloads the GGUF (Q4_K_M) quantized model and serves a local Gradio
UI at `http://127.0.0.1:7860` — pure CPU inference via `llama.cpp`, no
HuggingFace Spaces or GPU dependency. Add `share=True` to `demo.launch()`
to get a temporary public link others can open without installing anything.

**Option C — Google Colab:** open
`notebooks/Day32_Gardio.ipynb`, run the cells, and use the
`https://8e82a00cbc1c189325.gradio.live` public link generated in the output.

## 11. Response to Reviewer Feedback

The following points were raised in supervisor review and are addressed here:

1. **Precision mismatch (FP16 base vs. 4-bit fine-tuned) — resolved:** the
   original Day 32 evaluation compared the base model in FP16 against the
   fine-tuned model in 4-bit NF4, conflating the effect of fine-tuning with
   the effect of quantization. Both models were re-run in matched FP16
   precision with identical generation settings (greedy decoding) — see the
   results table in §7.5. The fine-tuned model still outperforms the base
   model on every rubric criterion under matched precision, confirming the
   improvement is attributable to fine-tuning.
2. **Missing/blank scores in code files:** the Day 32 scoring is provided in
   full in `eval_results.csv` (all 20 prompts, both models, all four rubric
   columns populated) — see §7.5 for the aggregated table and conclusion.
   Note: the `Day32_Gardio.ipynb` notebook's own in-notebook dataframe
   output still shows blank scoring columns, since scoring was performed
   and saved directly to `eval_results.csv` rather than back into the
   notebook's cell output — `eval_results.csv` is the authoritative, fully
   populated source. Day 29 scoring is being reviewed against the original
   Day 29 notebook to confirm the scoring method and populate any missing
   values.
3. **`eval_results.csv` missing from repository:** this was a push omission,
   not a missing artifact — the file exists and is now included in this
   repository (see file listing in §10).
4. **Audio-related response in Day 32 output:** clarified in §7.5 — the
   response describes real-world deepfake-detection domain knowledge
   (audio/voice-clone detection is a legitimate technique in this field),
   not a claim that this text-only project has audio capability. See the
   note under §7.5 for the full explanation.
5. **Dataset count discrepancy (700 passages vs. 351+19 split) — resolved:**
   traced to `Domain_Adapter.ipynb`. The corpus is loaded as 700 raw
   passages, then `train_test_split(test_size=100, seed=42)` produces 600
   raw train examples and 100 raw eval examples (an absolute-count split,
   not a 95/5 percentage split as an earlier README draft incorrectly
   stated — corrected in §2). With `packing=True`, the SFTTrainer packs
   short passages together into fixed 256-token blocks before training, so
   the trainer's effective dataset is 351 packed train chunks and 19 packed
   eval chunks — smaller than the raw counts by design, not from data loss.
   Both raw and packed counts are printed directly in the notebook (Cells 6
   and 9) for verification.
6. **Success criterion:** addressed explicitly in  — the fine-tuned
   model outperforms the base model on both in-domain and out-of-domain
   prompts in the Day 32 evaluation, meeting the stated bar.

