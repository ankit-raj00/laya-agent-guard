# Laya Agent Guard demo API + static frontend.
# Run: uvicorn server:app --host 127.0.0.1 --port 8000   (Caddy in front terminates HTTPS)
import json
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import laya
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

REPO = "ankitkr225/laya-agent-guard-softce"
LABELS = ["benign", "prompt_injection", "jailbreak", "harmful_request"]
MAX_CHARS = 4000            # the model reads at most 512 tokens anyway
RATE_LIMIT = (20, 60)       # requests per window (seconds), per client IP
HERE = Path(__file__).parent

torch.set_num_threads(2)
agent = laya.load(REPO, device="cpu")
QUESTION = json.load(open(hf_hub_download(REPO, "questions.json")))
agent.predict_batch([{"prompt": "warm up"}], QUESTION)   # first call is slow; don't make a visitor pay for it

lock = threading.Lock()     # one inference at a time on the CPU
hits = defaultdict(deque)
stats = {"checks": 0, "blocked": 0, "started": time.time()}
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class CheckIn(BaseModel):
    text: str = Field(min_length=1)
    threshold: float = Field(0.5, ge=0.05, le=0.95)


def client_ip(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else req.client.host


def rate_limited(ip: str) -> bool:
    n, window = RATE_LIMIT
    q, now = hits[ip], time.time()
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= n:
        return True
    q.append(now)
    return False


@app.post("/api/check")
def check(body: CheckIn, req: Request):
    if rate_limited(client_ip(req)):
        raise HTTPException(429, "Too many requests. Please wait a minute.")
    text = body.text[:MAX_CHARS]
    with lock:
        t0 = time.perf_counter()
        res = agent.predict_batch([{"prompt": text}], QUESTION)[0]
        model_ms = (time.perf_counter() - t0) * 1000
    raw = res["answers"]["verdict"]["probabilities"]
    total = sum(float(raw[l]) for l in LABELS)
    probs = {l: float(raw[l]) / total for l in LABELS}
    p_attack = 1 - probs["benign"]
    blocked = p_attack >= body.threshold
    stats["checks"] += 1
    stats["blocked"] += int(blocked)
    return {
        "probs": probs,
        "p_attack": p_attack,
        "blocked": blocked,
        "label": max(LABELS[1:], key=probs.get) if blocked else "benign",
        "model_ms": round(model_ms, 1),
        "chars": len(text),
        "truncated": len(body.text) > MAX_CHARS,
    }


@app.get("/api/info")
def info():
    return {"model": REPO, "params": "421M", "device": "CPU (AWS m7i-flex.large, 2 vCPU)",
            "checks_served": stats["checks"], "blocked": stats["blocked"]}


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")
