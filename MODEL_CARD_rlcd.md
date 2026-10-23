---
license: cc-by-nc-4.0
base_model: convaiinnovations/laya
language:
- en
pipeline_tag: text-classification
library_name: laya
tags:
- guardrail
- prompt-injection
- jailbreak-detection
- ai-safety
- agents
---

# Laya Agent Guard (RLCD)

This is the RLCD-objective variant of [**Laya Agent Guard**](https://huggingface.co/ankitkr225/laya-agent-guard-softce). It uses the same data, question and settings, and only the training loss differs (Laya's default RLCD objective instead of soft cross-entropy). The full documentation (usage, training data, per-benchmark results, limitations) is on the [main model card](https://huggingface.co/ankitkr225/laya-agent-guard-softce).

| Overall (18,617 test prompts, threshold 0.5) | RLCD (this model) | soft-CE |
|---|---|---|
| Attack-detection F1 | 0.878 | **0.883** |
| AUROC | 0.964 | **0.968** |
| Safe prompts wrongly blocked | **6.2%** | 7.2% |
| Four-class accuracy / macro F1 | **90.8% / 0.849** | 90.3% / 0.847 |
| Benchmarks won vs Llama Guard 3 1B | **10 of 11** | 9 of 11 |
| Fitted temperature | 4.80 | 1.76 |
| Median latency (T4) | 34 ms | 34 ms |

The RLCD model's raw scores are much more overconfident, so calibration needed a temperature of 4.80 against 1.76 for soft-CE. After calibration, both give usable probabilities. On this data the two objectives are effectively tied.

## How to use

The same as the main model, with `repo = "ankitkr225/laya-agent-guard-rlcd"`:

```python
import json, laya
from huggingface_hub import hf_hub_download

repo = "ankitkr225/laya-agent-guard-rlcd"
agent = laya.load(repo, device="cuda")
question = json.load(open(hf_hub_download(repo, "questions.json")))
probs = agent.predict_batch([{"prompt": "your text"}], question)[0]["answers"]["verdict"]["probabilities"]
```

**License:** CC BY-NC 4.0 (part of the training data is non-commercial). Base model: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (Apache-2.0).
