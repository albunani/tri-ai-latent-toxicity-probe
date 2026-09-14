# Debugging Notes

Detailed write-ups of the bugs found and fixed during development. These are
referenced from the main `README.md` but kept separate since the README has
a 6,000-character cap.

## 1. Wrong text column (WildJailbreak)

Initial extraction pulled from the `completion` column (the model's
refusal/response) instead of the correct prompt column (`vanilla` or
`adversarial`, selected per-row by `data_type`). Caught by manually
inspecting "toxic"-labeled samples and finding polite refusals instead of
harmful requests.

## 2. Wrong label column (ToxicChat)

`human_annotation` is a review-status flag, not a toxicity label. The real
label, `toxicity`, was found by cross-tabulating the two columns and
discovering `human_annotation` was uncorrelated with the dataset's own
toxicity scores in the way a label should be.

## 3. Off-by-one layer bug

A forward hook registered on `model.model.layers[14]` does not correspond to
`hidden_states[14]` — `hidden_states[0]` is the embedding output, so
`hidden_states[i]` is the output of `layers[i-1]`. This was the dominant
cause of the initial 0.57–0.60 Codabench scores despite ~0.94 local accuracy;
local accuracy was self-consistent (same wrong layer used throughout) and
did not expose the mismatch until compared against the platform's real
extraction.

## 4. Truncation-length mismatch

Initial extraction used `max_length=256`; the competition's actual recipe
truncates at 64 tokens. Verified empirically (not just by matching the
stated spec) via train/eval cross-comparison on held-out data — training at
64 tokens outperformed training at 256 tokens when evaluated at 64 tokens,
confirming the eval-matching truncation was the right choice.

## 5. scikit-learn version mismatch

A saved pipeline failed to load correctly on the Codabench platform due to a
local/platform sklearn version difference (the platform runs an older
version than the one Colab installs by default). `HistGradientBoostingClassifier`
in particular stores a NumPy `Generator`/`BitGenerator` object as part of its
fitted state, which does not unpickle reliably across version boundaries the
way plain arrays and linear-model coefficients do. Resolved by pinning the
local environment's scikit-learn version to match the platform before
retraining (see `requirements.txt`).
