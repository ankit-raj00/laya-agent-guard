# 03b - Control experiment: a plain ModernBERT-large sequence classifier on the same data.
# Laya's encoder is ModernBERT-large, so this isolates what Laya's decision head and pretraining add.
import os, glob, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # single GPU: DataParallel + ModernBERT's unpadding ran at 85 s/step and diverged
os.system("pip install -q 'transformers>=4.48,<5' 'accelerate<2' datasets huggingface_hub scikit-learn")

import numpy as np, pandas as pd, torch
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,
                          DataCollatorWithPadding, TrainerCallback)

DATA_REPO = "ankitkr225/agent-guardrail-data"
OUT_REPO = "ankitkr225/modernbert-large-agent-guard"
BASE = "answerdotai/ModernBERT-large"
LABELS = ["benign", "prompt_injection", "jailbreak", "harmful_request"]
SMOKE = False


def get_hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    for p in glob.glob("/kaggle/input/**/hf_token.txt", recursive=True):
        return open(p).read().strip()
    return None


HF_TOKEN = get_hf_token()
dl = lambda f: hf_hub_download(DATA_REPO, f, repo_type="dataset", token=HF_TOKEN)
train = pd.read_parquet(dl("train.parquet"))
val = pd.read_parquet(dl("val.parquet"))
if SMOKE:
    train, val = train.sample(1500, random_state=0), val.sample(300, random_state=0)
l2i = {l: i for i, l in enumerate(LABELS)}

tok = AutoTokenizer.from_pretrained(BASE)
def enc(df):
    ds = Dataset.from_dict({"text": df.text.tolist(), "label": [l2i[l] for l in df.label]})
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=512), batched=True, remove_columns=["text"])
train_ds, val_ds = enc(train), enc(val)

model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=4, id2label=dict(enumerate(LABELS)), label2id=l2i,
    reference_compile=False, attn_implementation="sdpa")


def metrics(p):
    logits, y = p
    pred = logits.argmax(-1)
    prob = torch.softmax(torch.tensor(logits), -1).numpy()
    yb, pb = (y != 0).astype(int), (1 - prob[:, 0] >= 0.5).astype(int)
    return {"macro_f1": f1_score(y, pred, average="macro"), "binary_f1": f1_score(yb, pb)}


args = TrainingArguments(
    output_dir="/kaggle/working/mb_out", num_train_epochs=1 if SMOKE else 2, learning_rate=2e-5,
    per_device_train_batch_size=8, per_device_eval_batch_size=32, gradient_accumulation_steps=8,
    warmup_ratio=0.06, weight_decay=0.01, gradient_checkpointing=True, lr_scheduler_type="cosine", fp16=True,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
    metric_for_best_model="binary_f1", group_by_length=True, logging_steps=20, report_to="none", seed=42,
    dataloader_num_workers=2)


class TimeBudget(TrainerCallback):
    """Stop before Kaggle's 12 h limit so the model still gets evaluated and pushed."""
    def __init__(self, hours): self.t0, self.limit = time.time(), hours * 3600
    def on_step_end(self, args, state, control, **kw):
        if state.global_step % 20 == 0:
            el = time.time() - self.t0
            print(f"step {state.global_step}/{state.max_steps} {el / max(state.global_step, 1):.1f}s/step", flush=True)
        if time.time() - self.t0 > self.limit:
            print("time budget reached at step", state.global_step, flush=True)
            control.should_training_stop = True


trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics,
                  callbacks=[TimeBudget(9.5)])
trainer.train()
res = trainer.evaluate()
print("final val:", res)

trainer.save_model("/kaggle/working/mb_final")
tok.save_pretrained("/kaggle/working/mb_final")
json.dump(res, open("/kaggle/working/mb_final/val_metrics.json", "w"), indent=2)
if HF_TOKEN and not SMOKE:
    api = HfApi(token=HF_TOKEN)
    api.create_repo(OUT_REPO, private=True, exist_ok=True)
    api.upload_folder(folder_path="/kaggle/working/mb_final", repo_id=OUT_REPO, commit_message="modernbert-large guard")
    print("pushed", OUT_REPO)
