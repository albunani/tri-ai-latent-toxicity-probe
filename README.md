# Latent Probing for Toxicity Detection

**Team OKU** — TRI AI Saturdays Lagos, Cohort 10
Team leader: Thaabit Idris · Members: Ogwu Naomi Chukwufumnanya, Riyas Yusuf

Codabench competition: [Latent Probe Challenge - Toxicity Detection](https://www.codabench.org/competitions/17670/)
**Result: 0.76 accuracy, 2nd place, Testing Phase (unseen hidden test set)**

---

## Overview

Given a mean-pooled hidden-state embedding from layer 14 of `google/gemma-2-2b`,
predict whether the underlying text was toxic (1) or safe (0). This repo
contains our full pipeline: dataset sourcing, embedding extraction, probe
training, and the final Codabench submission.

No dataset was provided by the competition — sourcing, cleaning, and labeling
the training data was part of the challenge itself.

---

## Dataset

We evaluated six public sources, verifying each one's actual label column
before use rather than assuming documented names were correct (this caught
two serious bugs — see `docs/debugging_notes.md`).

| Source | Rows used | Role |
|---|---|---|
| [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat) | 5,082 | Balanced toxic/safe chat prompts |
| [onepaneai/harmful-prompts](https://huggingface.co/datasets/onepaneai/harmful-prompts) | 200 | Toxic-only supplement |
| [PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | 7,766 (deduped from 27,186) | Wide harm-category coverage |
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | 3,000 (sampled, safe-only) | Realistic, non-adversarial safe requests |
| [allenai/wildjailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) | excluded | Synthetic/template-heavy; hurt generalization |
| [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) | excluded | Measures incivility, not harmful requests |

**Final training set: 4 sources, ~16,000 examples, balanced toxic/safe.**

### Why WildJailbreak and Civil Comments were excluded

Both were tested empirically, not excluded by assumption. WildJailbreak would
make up 94% of the combined pool if included, and every combination
containing it scored *lower* on the hidden test set than the same combination
without it (0.69 → 0.72 when removed) — its GPT-4/Mixtral-generated
adversarial prompts are stylistically synthetic compared to the other
sources. Civil Comments measures online incivility (profanity, rudeness)
rather than harmful *requests*, a different definition of "toxic," and
reduced accuracy in every combination tested (0.72→0.61 without WildJailbreak
present).

### Label handling notes

- **BeaverTails**: deduplicated repeated prompts; 32% of unique prompts had
  disagreeing labels across duplicates, resolved by majority vote.
- **ToxicChat**: the obvious label column (`human_annotation`) is actually a
  review-status flag, not a toxicity label — the real label is `toxicity`.
- **oasst1**: safe-class supplement only (English, `role=prompter`, Detoxify
  score ≤ 0.5); its small toxic-scoring subset was excluded after manual
  review found the score reflects crude language, not harmful intent.

---

## Training Pipeline

**Embedding extraction** (`src/extract_embeddings.py`): `google/gemma-2-2b`
(fp16); layer 14 via a forward hook on `model.model.layers[13]` (see
`docs/debugging_notes.md` for the off-by-one this avoids); mean pooling over
real tokens only (attention-mask-weighted); 64-token truncation, matching the
competition's extraction recipe.

**Classifier** (`src/train_probe.py`): Logistic Regression,
`class_weight='balanced'`, `C=0.1`; features standardized before fitting;
saved as a single `sklearn.pipeline.Pipeline` (scaler + classifier).

---

## Evaluation

| Configuration | Local held-out acc. | Codabench score |
|---|---|---|
| Layer bug present (wrong hook index) | ~0.94 (misleading) | 0.57–0.60 |
| 5 sources incl. WildJailbreak | 0.93 | 0.69 |
| 6 sources incl. Civil Comments | — | 0.65 |
| **4 sources, WildJailbreak + Civil Comments excluded, C=0.1** | **0.73** | **0.72 (Dev) / 0.76 (Testing)** |

The large local-vs-hidden-test gaps in early rows were the central debugging
problem of this project (see `docs/debugging_notes.md`). Once resolved, local
and hidden-test accuracy converged to a consistent, trustworthy signal.

---

## Reproduction

1. `pip install -r requirements.txt`
2. Set up Hugging Face access (`huggingface-cli login` or `HF_TOKEN` env var)
3. Run `src/extract_embeddings.py` — downloads the 4 sources, cleans/dedupes/
   labels each, extracts layer-14 embeddings, saves to disk
4. Run `src/train_probe.py` — trains and saves `trained_probe.joblib`
5. `src/classifier.py` is the Codabench-compatible submission interface; zip
   it with `trained_probe.joblib` at the root of the archive

---

## Limitations

This probe was trained and evaluated on **English-language data only**. The
project's original goal — extending coverage to Nigerian Pidgin, Hausa, and
Igbo via translation, with Yorùbá documented as a resource gap — was not
completed within the development window; development time went instead to
the debugging effort described above and to validating additional
English-language sources. We do not know how this probe performs on
non-English input. Full detail and reasoning are in `docs/data_card.pdf` and
`docs/impact_statement_card.pdf`.

---

## Appendix

**Contributors:** Thaabit Idris (team leader), Ogwu Naomi Chukwufumnanya,
Riyas Yusuf
**Mentors:** _[fill in mentor name(s)]_

Full write-ups of the five bugs found and fixed during development (wrong
text column, wrong label column, off-by-one layer bug, truncation-length
mismatch, scikit-learn version mismatch) are in `docs/debugging_notes.md`.
