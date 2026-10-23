# Evaluate guardrail models on the held-out test benchmarks (and val, for threshold tuning).
# Runs on Kaggle GPU. Results are pushed to the private HF dataset repo under results/<model>/.
import os, re, json, glob, time
os.environ["USE_TF"] = "0"
os.system("pip install -q -U laya==0.4.12 datasets huggingface_hub scikit-learn")

import numpy as np, pandas as pd, torch
from huggingface_hub import HfApi, hf_hub_download
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Which models this run evaluates (edited per push). Names are keys of MODEL_SPECS below.
MODELS = ["laya_ft_softce", "laya_ft_rlcd", "laya_base_guardpreset", "llama_guard3_1b", "prompt_guard2_86m", "protectai_deberta_v2"]  # add "jev" only with an OpenRouter key; it costs money

DATA_REPO = "ankitkr225/agent-guardrail-data"
OUT = "/kaggle/working/results"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


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
api = HfApi(token=HF_TOKEN)
dl = lambda f: hf_hub_download(DATA_REPO, f, repo_type="dataset", token=HF_TOKEN)
test = pd.read_parquet(dl("test.parquet"))
val = pd.read_parquet(dl("val.parquet"))
VERDICT_Q = json.load(open(dl("questions.json")))
LABELS = list(VERDICT_Q["verdict"]["criteria"].keys())
print("test rows:", len(test), "val rows:", len(val))


# ----------------------------------------------------------------------------- scorers
# Each scorer returns (p_unsafe: np.array, category_probs: np.array[N,4] or None)

def laya_scorer(model_id, mode):
    import laya
    from laya.integrations._guard import score_violation_probability
    agent = laya.load(model_id, device=DEV, token=HF_TOKEN)
    if mode == "guardpreset":
        q = laya.guard_questions()
        q = {k: v for k, v in q.items() if k in ("jailbreak", "prompt_injection", "harm_severity")}
    else:
        q = VERDICT_Q

    def run(texts):
        p_unsafe, cats = [], []
        for i in range(0, len(texts), 64):
            res = agent.predict_batch([{"prompt": t} for t in texts[i:i + 64]], q, batch_size=64, sort_by_length=True)
            for r in res:
                a = r["answers"]
                if mode == "guardpreset":
                    p_unsafe.append(max(a["jailbreak"]["noul"], a["prompt_injection"]["noul"],
                                        score_violation_probability(a["harm_severity"], 4)))
                else:
                    pr = a["verdict"]["probabilities"]
                    v = np.array([pr[l] for l in LABELS], dtype=float)
                    v = v / v.sum()
                    cats.append(v); p_unsafe.append(1.0 - v[0])
        return np.array(p_unsafe), (np.array(cats) if cats else None)

    def one(t):
        agent.predict_batch([{"prompt": t}], q)
    return run, one


def hf_multiclass_scorer(model_id):
    # 4-class classifier with LABELS order: p_unsafe = 1 - p(benign)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, token=HF_TOKEN).to(DEV).eval()
    order = [model.config.label2id[l] for l in LABELS]

    @torch.no_grad()
    def run(texts):
        cats = []
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], truncation=True, max_length=512, padding=True, return_tensors="pt").to(DEV)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                logits = model(**enc).logits.float()
            cats.append(torch.softmax(logits, -1)[:, order].cpu().numpy())
        cats = np.concatenate(cats)
        return 1.0 - cats[:, 0], cats

    @torch.no_grad()
    def one(t):
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            model(**tok([t], truncation=True, max_length=512, return_tensors="pt").to(DEV))
    return run, one


def hf_cls_scorer(model_id, unsafe_index_fn):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, token=HF_TOKEN).to(DEV).eval()
    if DEV == "cuda":
        model = model.half()
    idx = unsafe_index_fn(model.config)

    @torch.no_grad()
    def run(texts):
        out = []
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], truncation=True, max_length=512, padding=True, return_tensors="pt").to(DEV)
            out.append(torch.softmax(model(**enc).logits.float(), -1)[:, idx].cpu().numpy())
        return np.concatenate(out), None

    @torch.no_grad()
    def one(t):
        model(**tok([t], truncation=True, max_length=512, return_tensors="pt").to(DEV))
    return run, one


def llama_guard_scorer(model_id):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=HF_TOKEN, torch_dtype=torch.float16).to(DEV).eval()
    safe_id = tok.encode("safe", add_special_tokens=False)[0]
    unsafe_id = tok.encode("unsafe", add_special_tokens=False)[0]

    def prompt_ids(t):
        conv = [{"role": "user", "content": [{"type": "text", "text": t[:6000]}]}]
        out = tok.apply_chat_template(conv, return_tensors="pt", add_generation_prompt=True, return_dict=True)
        return out["input_ids"]

    @torch.no_grad()
    def score(t):
        ids = prompt_ids(t).to(DEV)
        logits = model(ids).logits[0, -1].float()
        # Llama Guard emits "\n\nsafe" / "\n\nunsafe"; step past leading newlines greedily
        for _ in range(3):
            nxt = int(logits.argmax())
            if nxt in (safe_id, unsafe_id):
                break
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], 1)
            logits = model(ids).logits[0, -1].float()
        two = torch.softmax(logits[[safe_id, unsafe_id]], -1)
        return float(two[1])

    def run(texts):
        return np.array([score(t) for t in texts]), None
    return run, score


def jev_scorer(model="~typesafe/jev-latest", budget_usd=0.95, workers=8):
    # Jev (TypeSafe) through OpenRouter's System One endpoint, asked exactly the same verdict question as Laya.
    # Every response carries usage.cost; calls stop for good once the spend reaches budget_usd.
    import requests, threading
    from concurrent.futures import ThreadPoolExecutor
    key = next(open(p).read().strip() for p in glob.glob("/kaggle/input/**/openrouter_key.txt", recursive=True))
    url = "https://openrouter.ai/api/v1/systemone"
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    lock, spent, fails = threading.Lock(), [0.0], [0]

    def call(t):
        for attempt in range(6):
            with lock:
                if spent[0] >= budget_usd:
                    raise RuntimeError(f"budget ${budget_usd} reached")
            try:
                r = requests.post(url, headers=hdr, timeout=60,
                                  json={"model": model, "state": {"prompt": t}, "questions": VERDICT_Q})
                if r.status_code == 200:
                    j = r.json()
                    with lock:
                        spent[0] += float(j.get("usage", {}).get("cost") or 0)
                    pr = j["answers"]["verdict"]["probabilities"]
                    v = np.array([float(pr.get(l, 0)) for l in LABELS]) + 1e-6
                    return v / v.sum()
                if r.status_code in (400, 401, 402, 403):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            except requests.RequestException:
                pass
            time.sleep(2 ** attempt)
        with lock:
            fails[0] += 1
        return np.array([0.25] * len(LABELS))

    def run(texts):
        with ThreadPoolExecutor(workers) as ex:
            cats = np.array(list(ex.map(call, texts)))
        print(f"jev: {len(texts)} calls, spent so far ${spent[0]:.4f}, failed {fails[0]}", flush=True)
        return 1.0 - cats[:, 0], cats

    def one(t):
        call(t)
    return run, one


def label_index(names):
    def fn(cfg):
        id2label = {int(k): str(v).lower() for k, v in cfg.id2label.items()}
        for i, n in id2label.items():
            if any(x in n for x in names):
                return i
        return max(id2label)
    return fn


MODEL_SPECS = {
    "laya_base_guardpreset": lambda: laya_scorer("convaiinnovations/laya", "guardpreset"),
    "laya_base_verdict": lambda: laya_scorer("convaiinnovations/laya", "verdict"),
    "protectai_deberta_v2": lambda: hf_cls_scorer("protectai/deberta-v3-base-prompt-injection-v2", label_index(["injection"])),
    "prompt_guard2_86m": lambda: hf_cls_scorer("meta-llama/Llama-Prompt-Guard-2-86M", label_index(["malicious", "label_1"])),
    "llama_guard3_1b": lambda: llama_guard_scorer("meta-llama/Llama-Guard-3-1B"),
    # fine-tuned checkpoints (private HF model repos)
    "laya_ft_rlcd": lambda: laya_scorer("ankitkr225/laya-agent-guard-rlcd", "verdict"),
    "laya_ft_softce": lambda: laya_scorer("ankitkr225/laya-agent-guard-softce", "verdict"),
    "modernbert_large_cls": lambda: hf_multiclass_scorer("ankitkr225/modernbert-large-agent-guard"),
    "jev": lambda: jev_scorer(),
}


# ----------------------------------------------------------------------------- metrics

def threshold_at_fpr(scores, y, fpr=0.02):
    neg = np.sort(scores[y == 0])
    return float(neg[int(np.ceil((1 - fpr) * len(neg))) - 1]) if len(neg) else 0.5


def bench_metrics(df, score_col, thr):
    out = {}
    for bench, g in list(df.groupby("benchmark")) + [("ALL", df)]:
        y = (g.label != "benign").astype(int).values
        s = g[score_col].values
        m = {"n": int(len(g)), "pos": int(y.sum())}
        for name, t in [("@0.5", 0.5), ("@val", thr)]:
            pred = (s >= t).astype(int)
            if 0 < y.sum() < len(y):
                m["f1" + name] = round(f1_score(y, pred), 4)
                m["precision" + name] = round(precision_score(y, pred, zero_division=0), 4)
                m["recall" + name] = round(recall_score(y, pred), 4)
                m["fpr" + name] = round(float(pred[y == 0].mean()), 4)
            else:
                m["flag_rate" + name] = round(float(pred.mean()), 4)   # all-benign: over-block; all-attack: detection
        if 0 < y.sum() < len(y):
            m["auroc"] = round(roc_auc_score(y, s), 4)
        out[bench] = m
    return out


def category_metrics(df):
    g = df[~df.binary_only & df.cat_pred.notna()]
    if not len(g):
        return None
    return {"macro_f1": round(f1_score(g.label, g.cat_pred, labels=LABELS, average="macro"), 4),
            "per_class_f1": dict(zip(LABELS, [round(x, 4) for x in f1_score(g.label, g.cat_pred, labels=LABELS, average=None)])),
            "accuracy": round(float((g.label == g.cat_pred).mean()), 4)}


# ----------------------------------------------------------------------------- run
summary = {}
for name in MODELS:
    print(f"\n######## {name}")
    try:
        run, one = MODEL_SPECS[name]()
    except Exception as e:
        print(f"SKIP {name}: {type(e).__name__}: {str(e)[:300]}")
        summary[name] = {"skipped": f"{type(e).__name__}: {str(e)[:200]}"}
        continue
    t0 = time.time()
    s_val, _ = run(val.text.tolist())
    thr = threshold_at_fpr(s_val, (val.label != "benign").astype(int).values, 0.02)
    s_test, cats = run(test.text.tolist())
    elapsed = time.time() - t0
    pred = test.copy()
    pred["p_unsafe"] = s_test
    pred["cat_pred"] = [LABELS[int(np.argmax(c))] for c in cats] if cats is not None else None
    if cats is not None:
        for j, l in enumerate(LABELS):
            pred["p_" + l] = cats[:, j]
    # latency at batch size 1 on 200 texts
    sample = test.text.sample(200, random_state=0).tolist()
    for t in sample[:10]:
        one(t)
    lat = []
    for t in sample:
        if DEV == "cuda": torch.cuda.synchronize()
        a = time.perf_counter(); one(t)
        if DEV == "cuda": torch.cuda.synchronize()
        lat.append((time.perf_counter() - a) * 1000)
    res = {"val_threshold_fpr2": round(thr, 4), "benchmarks": bench_metrics(pred, "p_unsafe", thr),
           "category": category_metrics(pred), "latency_ms_p50": round(float(np.median(lat)), 1),
           "latency_ms_p95": round(float(np.percentile(lat, 95)), 1), "eval_seconds": round(elapsed, 1)}
    summary[name] = res
    d = f"{OUT}/{name}"
    os.makedirs(d, exist_ok=True)
    pred.to_parquet(f"{d}/predictions.parquet", index=False)
    json.dump(res, open(f"{d}/metrics.json", "w"), indent=2)
    print(json.dumps({b: {k: v for k, v in m.items() if k in ("auroc", "f1@0.5", "recall@0.5", "fpr@0.5", "flag_rate@0.5")}
                      for b, m in res["benchmarks"].items()}, indent=1))
    print("category:", res["category"], "latency p50 ms:", res["latency_ms_p50"])
    del run, one
    torch.cuda.empty_cache()

json.dump(summary, open(f"{OUT}/summary_{'_'.join(MODELS)[:80]}.json", "w"), indent=2)
if HF_TOKEN:
    api.upload_folder(folder_path=OUT, path_in_repo="results", repo_id=DATA_REPO, repo_type="dataset",
                      commit_message="eval: " + ", ".join(MODELS))
    print("results pushed")
