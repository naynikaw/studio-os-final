# StudioOS Deployment Guide

## Architecture Overview

StudioOS consists of two parts:
- **Frontend**: React + Vite (deployed to Vercel)
- **Backend**: Python FastAPI (deployed to Render)

---

## 🚀 Backend Deployment (Render)

### Option A: One-Click Deploy with Blueprint

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **New** → **Blueprint**
3. Connect your GitHub repo
4. Select the `backend` folder
5. Render will auto-detect `render.yaml`

### Option B: Manual Deploy

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **New** → **Web Service**
3. Connect your GitHub repo
4. Configure:
   - **Name**: `studioos-backend`
   - **Root Directory**: `backend`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### Environment Variables (Required)

Set these in Render dashboard under **Environment**:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | Your OpenAI API key |
| `SERPAPI_KEY` | ✅ Yes | SerpAPI key for Reddit search |
| `NEWS_API_KEY` | ❌ Optional | For M6 news features |
| `ALPHAVANTAGE_API_KEY` | ❌ Optional | For M6 market data |

### Get Your Backend URL

After deployment, your backend URL will be:
```
https://studioos-backend.onrender.com
```
(or similar, based on your app name)

---

## 🌐 Frontend Deployment (Vercel)

### Step 1: Deploy to Vercel

1. Go to [Vercel](https://vercel.com)
2. Click **Add New** → **Project**
3. Import your GitHub repo
4. Configure:
   - **Root Directory**: `studio-os`
   - **Framework Preset**: Vite (should auto-detect)

### Step 2: Set Environment Variable

In Vercel project settings → **Environment Variables**:

| Variable | Value |
|----------|-------|
| `VITE_API_URL` | `https://studioos-backend.onrender.com` |

⚠️ Replace with your actual Render backend URL!

### Step 3: Redeploy

After adding the environment variable, trigger a new deployment.

---

## 🔧 Local Development

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
cd studio-os
npm install
npm run dev
```

---

## ⚠️ Important Notes

1. **Render Free Tier**: Backend sleeps after 15 min of inactivity. First request takes ~30 seconds to wake up.

2. **CORS**: Backend is configured to allow all origins. In production, you may want to restrict this.

3. **API Keys**: Never commit `.env` files with real API keys!
