# Model card — Braid (research preview)

This card describes the **Braid model family** served by this demo. The
authoritative, per-checkpoint card ships inside each model package as
`MODEL_CARD.md` and is generated from that package's manifest; this file is the
family-level version and is deliberately conservative.

## Summary

| field | value |
|---|---|
| name | Braid |
| developer | Solexsis Research |
| type | causal (autoregressive) language model over raw bytes |
| architecture | byte-level HourglassBraid — causal U-Net over the sequence |
| tokenizer | none; vocabulary is the 256 byte values |
| canonical presets | `default-300m` (295,082,179 params), `default-100m` (102,482,243 params) |
| context limit | 8,192 bytes (preset `block_size`) |
| status | **research preview — no trained public checkpoint yet** |
| licence (code) | Apache-2.0 |
| licence (weights) | recorded per package in `manifest.json` |

## Architecture

Braid replaces most transformer layers with a much cheaper local mixer and
spends attention only where it is needed:

* a **gated causal convolution** stem at full byte resolution,
* two **causal 4× downsamples** (16× total), each pooled sequence shifted one
  position so a coarse slot summarises only *completed* earlier groups,
* a shallow **dense-attention bottleneck** at the 1/16 rate, with a bounded
  attention window so the decode state is constant in context length,
* **upsampling with gated U-Net skips** back to byte resolution,
* one **256-byte sliding-window attention** block on the up path, which closes
  the recency gap the coarse path structurally cannot see,
* **hashed byte-n-gram (3,4) input embeddings** with prime buckets,
* an **exact 4-byte suffix-copy** branch mixed with the neural byte head,
* trained with **Muon** on 2D hidden matrices (AdamW elsewhere) under a
  warmup-stable-decay schedule, with multi-token-prediction auxiliary heads.

Each of those choices is backed by a measured A/B in the private research
ledger. The mechanisms that lost — a multi-resolution recency ladder, a
selective-convolution recency mixer, context-conditioned skip gates, deeper or
wider coarse bottlenecks, power-of-two n-gram hashing, learned dynamic chunking
— are not in the default model.

## Intended use

* Interactive text continuation, for demonstrating and evaluating the
  architecture.
* Research on tokenizer-free and hierarchical language models.
* Systems comparison against matched transformer baselines.

## Out of scope

* Factual question answering. It has no knowledge grounding and no retrieval.
* Code you intend to execute.
* Any decision affecting a person: medical, legal, financial, hiring,
  moderation, safety.
* Instruction following or chat. There is no instruction tuning.
* Long documents. The coarse attention window is bounded on purpose.

## Training data

The intended pretraining corpus is FineWeb-Edu as **raw bytes** with `0x00`
document separators. There is no tokenizer and no vocabulary file.

The exact byte count and dataset description for a given checkpoint are recorded
in that package's `manifest.json` and surfaced by `GET /v1/model`. When the demo
reports `"training_bytes": null`, no figure has been published — not zero.

## Evaluation

Published measurements are at **~30M parameters** on FineWeb bytes, three paired
seeds for the primary comparison, on an RTX 3090:

* Braid's full default recipe reaches **1.5296 ±0.0040** validation bits/byte;
  the matched dense MHA transformer reaches **1.5106 ±0.0070**.
* Braid trains at **3.07×** the steady-state throughput, with **2.23×** lower
  peak allocated memory and **2.99×** lower gross energy.

**The transformer wins equal-data quality.** Braid's advantage is cost. Anyone
telling you otherwise is reading a different table.

No 100M or 300M checkpoint of the current recipe has been trained, so no quality
number exists at those scales, and none is projected on the benchmark page.

## Limitations and failure modes

* Output is frequently incoherent, repetitive, or wrong.
* No alignment, no refusal behaviour, no safety filtering. It will continue
  whatever it is given.
* Byte-level generation can produce invalid UTF-8; the runtime buffers partial
  sequences and renders unrecoverable bytes as U+FFFD.
* The exact-copy branch makes literal repetition of recent text more likely,
  which is a real quality win on code and a visible artefact on prose.
* Bounded coarse attention limits long-range reach; the research repository's
  episodic-memory work is **not** enabled in the default model, because it was
  measured as bits-per-byte-neutral on web text.
* Biases present in web-scraped pretraining text are present in the model. No
  bias evaluation has been run.

## Ethical considerations

The model is served without moderation. Operators deploying it publicly are
responsible for adding whatever filtering their context requires. The demo does
not store prompts or completions, and does not log them.

## Provenance and reproducibility

Every package records the source commit of the private research repository, the
full resolved architecture configuration, the exact parameter count, the build
timestamp and SHA-256 hashes of every file. `braid inspect <package>` verifies
all of it. A package whose checksums, dtype, architecture options or runtime
range do not validate will not load.

## Citation

No paper yet. Cite the repository and the commit recorded in the package
manifest.
