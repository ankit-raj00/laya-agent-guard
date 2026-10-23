# Laya Agent Guard

A guardrail model for AI agents. It reads any text an agent is about to act on (user messages, emails, documents, web pages, tool results) and classifies it as **benign**, **prompt_injection**, **jailbreak** or **harmful_request** in about 34 ms, with calibrated probabilities.

It was built by fine-tuning [Laya](https://huggingface.co/convaiinnovations/laya), a 421M-parameter non-autoregressive decision model (ModernBERT-large + a small decision head), on 78k labelled examples from 13 public datasets plus synthetic indirect-injection data. Everything ran on free Kaggle GPUs.

**Models:** [laya-agent-guard-softce](https://huggingface.co/ankitkr225/laya-agent-guard-softce) (main) · [laya-agent-guard-rlcd](https://huggingface.co/ankitkr225/laya-agent-guard-rlcd)

![Headline comparison](assets/headline_comparison.png)

| Model | F1 | AUROC | Safe prompts wrongly blocked | Median latency |
|---|---|---|---|---|
| **Laya Agent Guard (soft-CE)** | **0.883** | **0.968** | 7.2% | 34 ms |
| **Laya Agent Guard (RLCD)** | 0.878 | 0.964 | **6.2%** | 34 ms |
| Jev (hosted API) | 0.782 | 0.924 | 17.2% | n/a (API, not comparable) |
| Llama Guard 3 1B | 0.649 | 0.809 | 18.7% | 98 ms |
| Laya base, no fine-tune | 0.578 | 0.694 | 34.2% | 44 ms |

18,617 test prompts, 11 benchmarks, threshold 0.5. Latency for the local models was measured on the same T4 GPU at batch size 1. The per-benchmark results, including where the model loses (it trails Jev on XSTest, JailbreakBench and deepset), are on the [model card](MODEL_CARD.md). The raw metrics for all 8 evaluated models are in [`results/`](results/).

## Pipeline

```
src/00_inspect_datasets.py         look at each candidate dataset's columns and labels
src/01_data_prep.py                build train / val / test with the label rules in LABEL_SPEC.md
src/02_finetune_laya.py            fine-tune Laya twice in parallel (RLCD on GPU 0, soft-CE on GPU 1), calibrate, push
src/02b_finetune_modernbert_control.py   control: plain ModernBERT-large classifier on the same data
src/03_eval_guard.py               evaluate any model on the 11 benchmarks (F1, AUROC, FPR, latency, 4-class report)
```

Each script runs as a Kaggle script kernel (GPU T4 ×2, internet on). Steps:
1. Accept the terms of the gated datasets on Hugging Face (WildGuardMix, XSTest, Qualifire) and of the gated baseline models (Llama Guard 3, Prompt Guard 2).
2. Add your Hugging Face token as a Kaggle secret named `HF_TOKEN`.
3. Change `HF_REPO` / `DATA_REPO` / output repo names to your own account.

The Jev baseline needs an OpenRouter key and costs money (about $0.50 for the full test set). It's off by default.

## How it works, briefly

Laya doesn't have a fixed output layer. The question and every answer option are written into the input, and each option gets a `[MASK]` "answer box":

```
[CLS] choice question: Classify `prompt` ... [SEP]
[MASK] benign: ordinary content ...  [MASK] prompt_injection: tries to override ...
[MASK] jailbreak: uses role-play ...  [MASK] harmful_request: directly asks for ... [SEP]
<the text to check> [SEP]
```

ModernBERT reads it all bidirectionally. Two extra Transformer layers let the boxes compare, and a shared scorer turns each box into one logit. A per-bucket temperature then makes the softmax calibrated. Because the labels are just text, they can be reworded or extended without changing the architecture. The question used here is in [`questions.json`](questions.json).

## Data and licences

The dataset isn't redistributed. Some sources are gated (WildGuardMix, XSTest, Qualifire), some are non-commercial (ToxicChat, Qualifire) and some have no stated licence (xTRam1 safe-guard, AESLC). `src/01_data_prep.py` rebuilds it after you accept each source's terms.

- **Code in this repo:** Apache-2.0.
- **Model weights:** CC BY-NC 4.0, because of the ToxicChat training data.

## Credits

[ConvAI Innovations](https://huggingface.co/convaiinnovations/laya) for Laya, [Answer.AI](https://huggingface.co/answerdotai/ModernBERT-large) for ModernBERT, and the authors of every dataset and benchmark used. See the model card for the full list.
