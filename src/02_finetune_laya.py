# 03 - Fine-tune Laya as an agent guardrail. Two runs in parallel, one per T4:
#   GPU0: loss="rlcd"   (Laya's default objective)
#   GPU1: loss="soft-ce" (soft cross-entropy only; matched or beat RLCD in Laya's own ablation)
# Each run calibrates temperatures on a held-out slice and is pushed to a private HF model repo.
import os, sys, json, glob, subprocess, time
os.environ["USE_TF"] = "0"
os.system("pip install -q -U laya==0.4.12 huggingface_hub")

DATA_REPO = "ankitkr225/agent-guardrail-data"
RUNS = {
    "rlcd": {"gpu": "0", "repo": "ankitkr225/laya-agent-guard-rlcd"},
    "soft-ce": {"gpu": "1", "repo": "ankitkr225/laya-agent-guard-softce"},
}
EPOCHS = 2
MICRO_BATCH = 16
GRAD_ACCUM = 4          # effective batch 64 per run, as in Laya's notebook
SMOKE = False           # True: 1,500 rows, 1 epoch, no push -- checks the pipeline end to end

WORKER = r'''
import os, sys, json
os.environ["USE_TF"] = "0"
import torch
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.train import TrainConfig, finetune
loss, out_dir, epochs, mb, ga = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
base = "/kaggle/working/laya_base"
if not os.path.exists(base + "/model.safetensors"):
    snapshot_download("convaiinnovations/laya", local_dir=base)
_fix_tokenizer_config(base)
cfg = TrainConfig(loss=loss, epochs=epochs, micro_batch=mb, grad_accum=ga, seed=42,
                  eval_data="/kaggle/working/data/val.jsonl", log_every=200)
summary = finetune("/kaggle/working/data/train.jsonl", base, out_dir, cfg, device="cuda")
json.dump(summary, open(out_dir + "/train_summary.json", "w"), indent=2, default=str)
print("DONE", loss, {k: summary.get(k) for k in ("train_items", "calibration_items", "temperature", "epoch_loss")})
'''


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
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
api = HfApi(token=HF_TOKEN)

os.makedirs("/kaggle/working/data", exist_ok=True)
for f in ["train.jsonl", "val.jsonl", "questions.json"]:
    p = hf_hub_download(DATA_REPO, f, repo_type="dataset", token=HF_TOKEN)
    os.system(f"cp {p} /kaggle/working/data/{f}")
if SMOKE:
    for f in ["train.jsonl", "val.jsonl"]:
        lines = open(f"/kaggle/working/data/{f}").readlines()[:1500 if f == "train.jsonl" else 300]
        open(f"/kaggle/working/data/{f}", "w").writelines(lines)
    EPOCHS = 1
print(open("/kaggle/working/data/train.jsonl").read().count("\n"), "train rows")

# download base once before forking so the two workers don't race
from laya.agent import _fix_tokenizer_config
snapshot_download("convaiinnovations/laya", local_dir="/kaggle/working/laya_base")
_fix_tokenizer_config("/kaggle/working/laya_base")
open("/kaggle/working/worker.py", "w").write(WORKER)

procs = {}
for loss, r in RUNS.items():
    out_dir = f"/kaggle/working/ckpt_{loss}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=r["gpu"])
    log = open(f"/kaggle/working/train_{loss}.log", "w")
    procs[loss] = (subprocess.Popen([sys.executable, "/kaggle/working/worker.py", loss, out_dir, str(EPOCHS),
                                     str(MICRO_BATCH), str(GRAD_ACCUM)], env=env, stdout=log, stderr=subprocess.STDOUT),
                   out_dir, log)

while any(p.poll() is None for p, _, _ in procs.values()):
    time.sleep(300)
    for loss in procs:
        tail = open(f"/kaggle/working/train_{loss}.log").read()[-400:].replace("\n", " | ")
        print(f"[{time.strftime('%H:%M')}] {loss}: {tail}", flush=True)

for loss, (p, out_dir, log) in procs.items():
    log.close()
    print(f"\n===== {loss} exit={p.returncode}")
    print(open(f"/kaggle/working/train_{loss}.log").read()[-3000:])
    src = out_dir if p.returncode == 0 else out_dir + "/checkpoint_latest"
    if HF_TOKEN and not SMOKE and os.path.exists(src + "/rl_agent_config.json"):
        q = "/kaggle/working/data/questions.json"
        os.system(f"cp {q} {src}/questions.json")
        repo = RUNS[loss]["repo"]
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=src, repo_id=repo, commit_message=f"laya guard fine-tune ({loss})",
                          ignore_patterns=["checkpoint_latest/*", "*.pt"])
        print("pushed", repo)
