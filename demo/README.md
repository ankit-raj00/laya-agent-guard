# Live demo

**https://guard.ankitrajai.in**: paste any text and see the verdict, the class probabilities and the timing. The page shows model time measured on the server and round-trip time measured in your browser.

## Files

| File | What it does |
|---|---|
| `server.py` | FastAPI app. It loads the model once and serves `POST /api/check`, which returns the probabilities, P(attack), the verdict and the model time in ms. Also serves `GET /api/info` and the page. Runs one inference at a time, limits each client IP to 20 requests a minute and caps input at 4,000 characters. |
| `index.html` | Single-file frontend (no build step, no external libraries): example prompts, a threshold slider, a verdict, probability bars, timing tiles, session history and benchmark results. Supports light and dark mode. |
| `deploy/guard-api.service` | systemd unit that runs uvicorn on `127.0.0.1:8000` and restarts it on failure |
| `deploy/Caddyfile.example` | Caddy reverse proxy with automatic HTTPS |

## Run it yourself

```bash
python -m venv venv && . venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "laya==0.4.12" "transformers>=4.48,<5" fastapi "uvicorn[standard]" huggingface_hub
uvicorn server:app --port 8000     # then open http://localhost:8000
```

The hosted demo runs on an AWS EC2 `m7i-flex.large` (2 vCPU, 8 GB, Ubuntu 24.04) in ap-south-1, with Caddy in front. There, a check takes about 1.2 s on the CPU. On a T4 GPU the same model takes about 34 ms.
