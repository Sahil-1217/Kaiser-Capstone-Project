# Kaiser Permanente Member Experience Intelligence Platform
# Production Deployment & Hosting Guide

This guide details how to deploy the **Kaiser Permanente Member Experience Intelligence Platform** to the cloud so anyone can access the live interactive application via a public HTTPS URL.

---

## Architecture Overview
- **Application Web Tier:** Streamlit Enterprise Cockpit (`app.py`) + Standalone Dynamic 6-Page Executive Dashboard (`docs/senior_standalone_dashboard.html`).
- **Relational Persistence Layer:** Neon Cloud PostgreSQL (`neondb` on AWS us-east-2) with encrypted SSL connection.
- **Machine Learning & NLP Engine:** Scikit-learn (Random Forest, Decision Tree, Logistic Regression, Gradient Boosting) + PyTorch 1D-CNN Sentiment Model (`models/pytorch_sentiment_model.pt`).
- **MediRAG Care Navigation Assistant:** Hybrid Vector/BM25 Triage Engine with deterministic Clinical Safety Gatekeeper.

---

## Method 1: Streamlit Community Cloud (Recommended — Free & 2 Minutes)

Streamlit Community Cloud is the fastest, free way to deploy directly from your GitHub repository (`Sahil-1217/Kaiser-Capstone-Project`).

### Step 1: Commit and Push Your Latest Changes to GitHub
From your terminal:
```bash
git add .
git commit -m "Deploy: Neon PostgreSQL live sync, 7-tab layout, and production Dockerfile"
git push origin main
```

### Step 2: Sign In to Streamlit Community Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)**.
2. Sign in with your **GitHub account**.

### Step 3: Create New App
1. Click **"Create app"** (or **"New app"**).
2. Choose **"I already have an app"**.
3. Fill in the deployment form:
   - **Repository:** `Sahil-1217/Kaiser-Capstone-Project`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** (Optional custom subdomain, e.g., `kaiser-member-intelligence.streamlit.app`)

### Step 4: Add Cloud Database Secrets
Before clicking Deploy, click **"Advanced settings..."** ➔ **"Secrets"**:
Paste your Neon PostgreSQL connection string:
```toml
DATABASE_URL = "postgresql+psycopg2://neondb_owner:npg_gY5y3XkWviLp@ep-rough-wave-b4tlujf7-pooler.c-6.us-east-2.aws.neon.tech/neondb?sslmode=require"
```
*(You can also add `OPENAI_API_KEY` here if you want live OpenAI LLM calls for MediRAG).*

### Step 5: Click "Deploy!"
Streamlit will install `requirements.txt`, connect to Neon Cloud PostgreSQL, and launch your live application with a public HTTPS link!

---

## Method 2: Docker Container (Render / Railway / Cloud Run)

The repository includes a production-ready `Dockerfile`.

### Local Testing with Docker:
```bash
# 1. Build the Docker container image
docker build -t kaiser-analytics .

# 2. Run the container
docker run -p 8501:8501 --env-file .env kaiser-analytics
```
Then open `http://localhost:8501` in your browser.

### Deploying to Render (Free Web Service):
1. Go to **[render.com](https://render.com)** and create a new **Web Service**.
2. Connect your GitHub repository: `Sahil-1217/Kaiser-Capstone-Project`.
3. Select **Docker** as the Environment.
4. Under **Environment Variables**, add:
   - `DATABASE_URL`: Your Neon connection string.
   - `PORT`: `8501`
5. Click **Create Web Service**.

### Deploying to Railway:
1. Go to **[railway.app](https://railway.app)**.
2. Click **New Project** ➔ **Deploy from GitHub repo**.
3. Select `Sahil-1217/Kaiser-Capstone-Project`.
4. Add the `DATABASE_URL` environment variable. Railway will auto-detect the `Dockerfile` and deploy.

---

## Method 3: Hugging Face Spaces (Free Streamlit Hosting)

1. Go to **[huggingface.co/spaces](https://huggingface.co/spaces)**.
2. Click **"Create new Space"**.
3. Choose **Streamlit** SDK.
4. Clone the space or link your GitHub repo.
5. In **Settings ➔ Variables and secrets**, add `DATABASE_URL`.
