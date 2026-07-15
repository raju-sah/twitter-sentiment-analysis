# Twitter Sentiment Analysis

A Twitter / text sentiment classifier built from scratch (logistic regression on
NLTK's `twitter_samples` corpus) exposed through a production-style web app:

- **Frontend:** React + Vite, deployed on [Vercel](https://vercel.com)
- **Backend:** FastAPI serving a trained model, deployed on [Render](https://render.com)
- **Model:** Logistic regression implemented from scratch (sigmoid + gradient descent)

Users type a tweet or sentence; the API returns a positive/negative label with a
confidence score and the processed tokens.

## Architecture

```
Browser (React + Vite @ Vercel)
        │  POST /predict  (HTTPS, CORS)
        ▼
FastAPI @ Render (free tier)
        │  loads model.pkl on startup
        ▼
{ freqs, theta }   ← trained offline, committed as backend/model.pkl
```

## Tech Stack

- Frontend: React 18, Vite 5
- Backend: FastAPI, Uvicorn, Pydantic
- ML: NumPy, NLTK (custom logistic regression)
- Deploy: Vercel (frontend) + Render (backend)

## Repository Layout

```
twitter-sentiment-analysis/
├── backend/            # FastAPI service → Render
│   ├── app.py          # API: /predict, /health, CORS
│   ├── train.py        # trains the model and writes model.pkl
│   ├── utils.py        # process_tweet, feature extraction, prediction
│   ├── nltk_setup.py   # ensures NLTK stopwords are available
│   ├── model.pkl       # trained weights (committed)
│   ├── requirements.txt
│   └── test_api.py
├── frontend/           # React + Vite → Vercel
│   ├── src/App.jsx
│   └── ...
├── render.yaml         # Render blueprint for the backend service
├── MT.ipynb            # original course notebook (untouched)
├── sentimentalAnalysis.ipynb
├── utils.py            # root copy for the notebooks
├── w1_unittest.py
└── README.md
```

## Local Setup

### Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (optional) retrain and regenerate model.pkl
python train.py

uvicorn app:app --reload --port 8000
```

The API is then available at `http://localhost:8000`. Try it:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I love this movie, it was fantastic!"}'
```

Run the tests with `pytest`.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env        # set VITE_API_URL to your backend URL
npm run dev                 # http://localhost:5173 (proxies /api -> :8000)
```

In dev, requests to `/api/predict` are proxied to `http://localhost:8000`, so you
can leave `VITE_API_URL` unset locally.

## Deployment

### Backend → Render

The repo includes a `render.yaml` blueprint, so the easiest path is:

1. Push this repo to GitHub.
2. In Render, choose **New → Blueprint** and pick this repo — it reads `render.yaml`
   and provisions the `twitter-sentiment-api` web service automatically.

Or configure a **Web Service** manually:

1. New → **Web Service** from the GitHub repo.
2. **Root directory:** `backend`
3. **Runtime:** Python
4. **Build command:** `pip install -r requirements.txt && python -m nltk.downloader stopwords`
5. **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. **Health check path:** `/health`

Render assigns a public `*.onrender.com` URL and injects `PORT`. `model.pkl` is
committed, so no training happens at deploy time. NLTK `stopwords` is downloaded
at build time.

> Note: Render's free tier sleeps after ~15 min of inactivity, so the first
> request after idle can take 20–60s to wake up. Fine for a portfolio demo.

### Frontend → Vercel

1. Import the GitHub repo into Vercel.
2. Set the **root directory** to `frontend`.
3. Framework preset: **Vite** (build `npm run build`, output `dist`).
4. Add environment variable `VITE_API_URL` = your Render backend URL.
5. Deploy. The frontend calls `${VITE_API_URL}/predict`.

## Notes on Accuracy

The model reaches ~**99.5% accuracy** on the NLTK `twitter_samples` *test split*,
which is the same distribution it was trained on (the `freqs` dictionary is built
from the training tweets). On genuinely novel, modern text the accuracy is lower,
and the model struggles with sarcasm and negation (e.g. "not bad" → negative) — a
known limitation of unigram frequency features. This is expected for a from-scratch
logistic-regression baseline and is honestly reflected here rather than overstated.

## License

MIT
