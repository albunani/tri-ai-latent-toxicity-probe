"""
extract_embeddings.py

Loads the 4 source datasets, cleans/dedupes/labels each correctly, and
extracts Gemma-2-2B layer-14 mean-pooled embeddings for every example.

Usage: python extract_embeddings.py
Requires: a CUDA GPU, HF_TOKEN env var (or prior `huggingface-cli login`)
Output: <OUTPUT_DIR>/<source>_embeddings.npy and <source>_labels.npy for
        each of: toxic, harmfulprompts, beavertails, oasst1
"""

import os
import gc
import random
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "google/gemma-2-2b"
LAYER = 14  # target hidden_states[14]
MAX_LENGTH = 64  # matches the competition's stated extraction recipe
BATCH_SIZE = 8
SAVE_EVERY = 200  # batches per checkpoint file


# ============================================================
# Dataset loading + cleaning
# ============================================================
def load_sources():
    print("Loading ToxicChat...")
    toxic = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    toxic_texts = list(toxic["train"]["user_input"])
    # NOTE: 'human_annotation' looks like a label but is actually a review-status
    # flag. The real toxicity label is 'toxicity'. See README Appendix.
    toxic_labels = list(toxic["train"]["toxicity"])

    print("Loading onepaneai/harmful-prompts...")
    hp = load_dataset("onepaneai/harmful-prompts")
    hp_texts = list(hp["train"]["prompt"])
    hp_labels = [1] * len(hp_texts)  # toxic-only dataset, no safe class

    print("Loading BeaverTails...")
    bt = load_dataset("PKU-Alignment/BeaverTails", "default")
    bt_train = bt["30k_train"]
    bt_df = pd.DataFrame({
        "prompt": list(bt_train["prompt"]),
        "label": [0 if s else 1 for s in bt_train["is_safe"]],  # is_safe polarity is inverted vs. our convention
    })
    # Prompts repeat across multiple response/category rows; dedupe with
    # majority-vote label resolution for conflicting duplicates.
    deduped = bt_df.groupby("prompt")["label"].apply(lambda g: g.mode()[0]).reset_index()
    bt_texts = deduped["prompt"].tolist()
    bt_labels = deduped["label"].tolist()

    print("Loading oasst1...")
    oasst = load_dataset("OpenAssistant/oasst1")
    oasst_safe = [
        row for row in oasst["train"]
        if row["role"] == "prompter" and row["lang"] == "en"
        and row["detoxify"] is not None and row["detoxify"]["toxicity"] <= 0.5
    ]
    oasst_texts = [r["text"] for r in oasst_safe]
    random.seed(42)
    if len(oasst_texts) > 3000:
        idx = random.sample(range(len(oasst_texts)), 3000)
        oasst_texts = [oasst_texts[i] for i in idx]
    oasst_labels = [0] * len(oasst_texts)

    return {
        "toxic": (toxic_texts, toxic_labels),
        "harmfulprompts": (hp_texts, hp_labels),
        "beavertails": (bt_texts, bt_labels),
        "oasst1": (oasst_texts, oasst_labels),
    }


# ============================================================
# Model + corrected extraction hook
# ============================================================
def setup_model():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to("cuda")
    model.eval()

    captured = {}

    def _capture(module, input, output):
        captured["layer_output"] = output[0] if isinstance(output, tuple) else output

    # hidden_states[0] is the embedding output; hidden_states[i] is the
    # output of layers[i-1] for i>=1. To match hidden_states[LAYER], hook
    # layers[LAYER - 1], NOT layers[LAYER].
    model.model.layers[LAYER - 1].register_forward_hook(_capture)
    return model, tokenizer, captured


def masked_mean_pool(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def extract_embeddings(texts, model, tokenizer, captured, save_prefix):
    total = len(texts)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    chunk = []
    batch_idx = 0
    while batch_idx < total_batches:
        cs = (batch_idx // SAVE_EVERY) * SAVE_EVERY
        ce = min(cs + SAVE_EVERY - 1, total_batches - 1)
        chunk_file = os.path.join(OUTPUT_DIR, f"{save_prefix}_batch_{cs}_to_{ce}.npy")
        if os.path.exists(chunk_file):
            batch_idx = ce + 1
            continue
        i = batch_idx * BATCH_SIZE
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs, use_cache=False)
            pooled = masked_mean_pool(captured["layer_output"], inputs["attention_mask"])
            chunk.append(pooled.cpu().numpy())
        captured.clear()
        del inputs
        if batch_idx == ce or batch_idx == total_batches - 1:
            np.save(chunk_file, np.concatenate(chunk))
            print(f"  {save_prefix}: {min((batch_idx+1)*BATCH_SIZE, total)}/{total}")
            chunk = []
            gc.collect()
            torch.cuda.empty_cache()
        batch_idx += 1


def stitch_chunks(save_prefix):
    files = sorted(
        [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"{save_prefix}_batch_")],
        key=lambda f: int(f.split("_batch_")[1].split("_to_")[0]),
    )
    return np.concatenate([np.load(os.path.join(OUTPUT_DIR, f)) for f in files])


if __name__ == "__main__":
    sources = load_sources()
    model, tokenizer, captured = setup_model()

    for name, (texts, labels) in sources.items():
        print(f"--- Extracting {name} ({len(texts)} rows) ---")
        extract_embeddings(texts, model, tokenizer, captured, save_prefix=name)
        emb = stitch_chunks(name)
        assert emb.shape[0] == len(texts), f"{name}: row count mismatch after extraction"
        np.save(os.path.join(OUTPUT_DIR, f"{name}_embeddings.npy"), emb)
        np.save(os.path.join(OUTPUT_DIR, f"{name}_labels.npy"), np.array(labels))
        print(f"  Saved {name}_embeddings.npy {emb.shape}")

    print("Done.")
