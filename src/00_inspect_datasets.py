# Inspect candidate guardrail datasets: configs, splits, columns, sizes, label distributions, samples.
# Runs on Kaggle (CPU). Writes /kaggle/working/inspect_report.json and prints a readable summary.
import json, os, collections, traceback
os.system("pip install -q -U datasets huggingface_hub")

from datasets import load_dataset, load_dataset_builder, get_dataset_config_names, get_dataset_split_names

HF_TOKEN = None
SECRET_STATUS = "not loaded"
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    SECRET_STATUS = f"loaded, starts with {HF_TOKEN[:3]}, length {len(HF_TOKEN)}"
except Exception as e:
    SECRET_STATUS = f"{type(e).__name__}: {e}"
# Fallback: token stored in the owner's private Kaggle dataset
if not HF_TOKEN:
    import glob
    for p in glob.glob("/kaggle/input/**/hf_token.txt", recursive=True):
        HF_TOKEN = open(p).read().strip()
        SECRET_STATUS = f"loaded from private dataset, length {len(HF_TOKEN)}"
        break
print("HF_TOKEN secret status:", SECRET_STATUS)

DATASETS = [
    "allenai/wildguardmix",
    "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
    "xTRam1/safe-guard-prompt-injection",
    "deepset/prompt-injections",
    "Lakera/gandalf_ignore_instructions",
    "Lakera/mosscap_prompt_injection",
    "reshabhs/SPML_Chatbot_Prompt_Injection",
    "jackhhao/jailbreak-classification",
    "microsoft/llmail-inject-challenge",
    "bench-llm/or-bench",
    "ToxicityPrompts/PolyGuardMix",
    "lmsys/toxic-chat",
    "databricks/databricks-dolly-15k",
    "glaiveai/glaive-function-calling-v2",
    "JailbreakBench/JBB-Behaviors",
    "walledai/XSTest",
    "qualifire/prompt-injections-benchmark",
]
# Re-run only the gated datasets (others already inspected in version 1)
DATASETS = ["allenai/wildguardmix", "walledai/XSTest", "qualifire/prompt-injections-benchmark"]
SAMPLE_ROWS = 3000  # rows streamed per split to estimate label distributions
LABEL_MAX_UNIQUE = 30


def short(v, n=300):
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s[:n]


def inspect_split(name, config, split):
    out = {}
    try:
        b = load_dataset_builder(name, config, token=HF_TOKEN)
        if b.info.splits and split in b.info.splits:
            out["num_rows"] = b.info.splits[split].num_examples
    except Exception:
        pass
    ds = load_dataset(name, config, split=split, streaming=True, token=HF_TOKEN)
    rows = list(ds.take(SAMPLE_ROWS))
    out["sampled"] = len(rows)
    out["columns"] = list(rows[0].keys()) if rows else []
    dists = {}
    for col in out["columns"]:
        vals = [r[col] for r in rows]
        if all(isinstance(v, (str, int, float, bool, type(None))) for v in vals):
            c = collections.Counter(vals)
            if len(c) <= LABEL_MAX_UNIQUE:
                dists[col] = {str(k): v for k, v in c.most_common()}
    out["label_like_distributions"] = dists
    out["examples"] = [{k: short(v) for k, v in r.items()} for r in rows[:2]]
    return out


report = {"_secret_status": SECRET_STATUS}
for name in DATASETS:
    print(f"\n=== {name}")
    entry = {}
    try:
        try:
            configs = get_dataset_config_names(name, token=HF_TOKEN)
        except Exception:
            configs = [None]
        for config in configs:
            splits = get_dataset_split_names(name, config, token=HF_TOKEN)
            for split in splits:
                key = f"{config}/{split}"
                try:
                    entry[key] = inspect_split(name, config, split)
                    e = entry[key]
                    print(f"  [{key}] rows={e.get('num_rows')} cols={e['columns']}")
                    for col, d in e["label_like_distributions"].items():
                        print(f"     {col}: {d}")
                except Exception as ex:
                    entry[key] = {"error": f"{type(ex).__name__}: {str(ex)[:300]}"}
                    print(f"  [{key}] ERROR {entry[key]['error']}")
    except Exception as ex:
        entry["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
        print("  ERROR", entry["error"])
    report[name] = entry

with open("/kaggle/working/inspect_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)
print("\nSaved /kaggle/working/inspect_report.json")
