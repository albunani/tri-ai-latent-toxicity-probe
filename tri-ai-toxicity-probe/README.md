# Latent Probing for Toxicity Detection

**Team OKU** — TRI AI Saturdays Lagos, Cohort 10
Team leader: Thaabit Idris · Members: Ogwu Naomi Chukwufumnanya, Riyas Yusuf

Codabench competition: [Latent Probe Challenge - Toxicity Detection](https://www.codabench.org/competitions/17670/)
**Result: 0.76 accuracy, 2nd place, Testing Phase (unseen hidden test set)**

---

## Overview

Given a mean-pooled hidden-state embedding from layer 14 of `google/gemma-2-2b`,
predict whether the underlying text was toxic (1) or safe (0). This repository
contains our full pipeline: dataset sourcing, embedding extraction, probe
training, and the final Codabench submission.

No dataset was provided by the competition — sourcing, cleaning, and labeling
the training data was part of the challenge itself.

---

## Dataset

We evaluated **six public sources**, verifying each one's actual label
column before use rather than assuming documented column names were correct
(see Appendix — this caught two serious bugs).

| Source | Rows used | Role |
|---|---|---|
| [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat) | 5,082 | Balanced toxic/safe chat prompts |
| [onepaneai/harmful-prompts](https://huggingface.co/datasets/onepaneai/harmful-prompts) | 200 | Toxic-only supplement (jailbreak/injection styles) |
| [PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | 7,766 (deduped from 27,186) | Balanced, wide harm-category coverage |
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | 3,000 (sampled, safe-only) | Realistic, non-adversarial safe requests |
| [allenai/wildjailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) | — (excluded) | Synthetic/template-heavy; empirically hurt generalization (see Evaluation) |
| [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) | — (excluded) | Different toxicity definition (incivility vs. harmful requests); hurt every combination tried |

**Final training set: 4 sources, ~16,000 examples**, balanced toxic/safe.

### Why WildJailbreak and Civil Comments were excluded

Both were tested empirically, not excluded by assumption:

- **WildJailbreak** makes up 261,538 of a possible ~277,000 combined rows (94%)
  if included. Every combination that included it scored *lower* on the hidden
  test set than the same combination without it (0.69 → 0.72 when removed).
  Its prompts are GPT-4/Mixtral-generated adversarial jailbreaks — stylistically
  synthetic and template-heavy compared to the other sources.
- **Civil Comments** measures online incivility (profanity, rudeness), not
  harmful *requests* — a different definition of "toxic" than the other
  datasets use. It reduced accuracy in both combinations tested (with
  WildJailbreak: 0.69→0.65; without: 0.72→0.61).

### Label handling notes

- **BeaverTails**: prompts repeat across multiple response/category rows;
  deduplicated to unique prompts, resolving label conflicts (32% of unique
  prompts had disagreeing labels across duplicates) via majority vote.
- **ToxicChat**: the obvious label column (`human_annotation`) is actually a
  "was this reviewed" flag, not a toxicity label — the real label is
  `toxicity`. Verified via cross-tabulation before use (see Appendix).
- **oasst1**: used only as a safe-class supplement (English, `role=prompter`,
  Detoxify score ≤ 0.5). Its small toxic-scoring subset was excluded — manual
  inspection showed Detoxify's score reflects crude/explicit language, not
  harmful-request intent, and did not match this task's definition of toxic.

---

## Training Pipeline

**Embedding extraction** (`src/extract_embeddings.py`):
- Model: `google/gemma-2-2b`, loaded in fp16
- Layer: 14 (via a forward hook on `model.model.layers[13]` — see Appendix
  for why `layers[14]` is the wrong index)
- Pooling: mean over real tokens only (attention-mask-weighted, padding
  excluded)
- Truncation: 64 tokens (matches the competition's stated extraction recipe)
- `use_cache=False` (no autoregressive KV-cache needed for a single forward pass)

**Classifier** (`src/train_probe.py`):
- Logistic Regression, `class_weight='balanced'`, `C=0.1`
- Features standardized (`StandardScaler`) before fitting
- Saved as a single `sklearn.pipeline.Pipeline` (scaler + classifier), matching
  the starter kit's documented `joblib.dump(your_model, ...)` pattern

---

## Evaluation

| Configuration | Local held-out accuracy | Codabench score |
|---|---|---|
| Layer bug present (wrong hook index) | ~0.94 (self-consistent, misleading) | 0.57–0.60 |
| 5 sources incl. WildJailbreak | 0.93 | 0.69 |
| 6 sources incl. Civil Comments | — | 0.65 |
| **4 sources, WildJailbreak + Civil Comments excluded, C=0.1** | **0.73** | **0.72 (Development) / 0.76 (Testing, unseen set)** |

The large local-vs-hidden-test gaps in early rows were the central debugging
problem of this project (see Appendix). Once resolved, local and hidden-test
accuracy converged to a consistent, trustworthy signal.

---

## Reproduction

1. `pip install -r requirements.txt`
2. Set up Hugging Face access (`huggingface-cli login` or `HF_TOKEN` env var)
3. Run `src/extract_embeddings.py` — downloads the 4 source datasets, cleans/
   dedupes/labels each, extracts layer-14 embeddings, saves to disk
4. Run `src/train_probe.py` — trains and saves `trained_probe.joblib`
5. `src/classifier.py` is the Codabench-compatible submission interface;
   zip it together with `trained_probe.joblib` at the root of the archive

---

## Appendix

### Contributors
- Thaabit Idris (team leader)
- Ogwu Naomi Chukwufumnanya
- Riyas Yusuf

### Mentors
_[fill in mentor name(s)]_

### Bugs found and fixed during development

1. **Wrong text column (WildJailbreak):** initial extraction pulled from the
   `completion` column (the model's refusal/response) instead of the correct
   prompt column (`vanilla` or `adversarial`, selected per-row by `data_type`).
   Caught by manually inspecting "toxic"-labeled samples and finding polite
   refusals instead of harmful requests.
2. **Wrong label column (ToxicChat):** `human_annotation` is a review-status
   flag, not a toxicity label. The real label, `toxicity`, was found by
   cross-tabulating the two columns and discovering `human_annotation` was
   uncorrelated with the dataset's own toxicity scores in the way a label
   should be.
3. **Off-by-one layer bug:** a forward hook registered on
   `model.model.layers[14]` does not correspond to `hidden_states[14]` —
   `hidden_states[0]` is the embedding output, so `hidden_states[i]` is the
   output of `layers[i-1]`. This was the dominant cause of the initial
   0.57–0.60 Codabench scores despite ~0.94 local accuracy; local accuracy
   was self-consistent (same wrong layer used throughout) and did not expose
   the mismatch until compared against the platform's real extraction.
4. **Truncation-length mismatch:** initial extraction used `max_length=256`;
   the competition's actual recipe truncates at 64 tokens. Verified
   empirically (not just by matching the stated spec) via train/eval
   cross-comparison on held-out data — training at 64 tokens outperformed
   training at 256 tokens when evaluated at 64 tokens, confirming the
   eval-matching truncation was the right choice.
5. **scikit-learn version mismatch:** a saved pipeline failed to load
   correctly on the Codabench platform due to a local/platform sklearn
   version difference; resolved by pinning the version (see
   `requirements.txt`).
