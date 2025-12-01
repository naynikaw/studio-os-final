# Module Deployment Guide

Each module (M2-M6) is deployed as an **independent microservice** on Railway. This allows each module to be maintained and deployed separately.

## Module Structure

```
modules/
├── m2/
│   ├── app.py          # FastAPI app for M2
│   └── railway.json    # Railway deployment config
├── m3/
│   ├── app.py
│   └── railway.json
├── m4/
│   ├── app.py
│   └── railway.json
├── m5/
│   ├── app.py
│   └── railway.json
└── m6/
    ├── app.py
    └── railway.json
```

## Standardized API Contract

All modules follow the same API contract:

### Endpoint
`POST /api/process`

### Request
```json
{
  "prompt": "Your problem statement here"
}
```

### Response
```json
{
  "success": true,
  "data": {
    // Module-specific output
  },
  "message": null
}
```

### Health Check
`GET /health` - Returns service status

## Deploying Each Module

### Step 1: Create Railway Project

For each module, create a new Railway project:
- Go to Railway dashboard
- Click "New Project"
- Select "Deploy from GitHub repo"
- Choose your repository

### Step 2: Configure Service

1. **Set Root Directory** (if needed):
   - For M2: `modules/m2`
   - For M3: `modules/m3`
   - etc.

2. **Set Environment Variables in Railway Dashboard:**

   **How to set environment variables in Railway:**
   1. Open your service in Railway dashboard
   2. Go to the "Variables" tab
   3. Click "New Variable"
   4. Add each variable below

   **M2 (Problem Validation):**
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=your_reddit_client_secret
   REDDIT_USER_AGENT=StudioOS-M2/1.0 (by u/yourusername)
   PORT=8000
   ```
   
   **Where to get these keys:**
   - Gemini API: https://aistudio.google.com/app/apikey
   - Reddit API: https://www.reddit.com/prefs/apps → Create App (type: script)

   **M3 (Problem Understanding):**
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   BEA_API_KEY=your_bea_key_here (optional - for economic data)
   PORT=8000
   ```
   
   **Where to get these keys:**
   - Gemini API: https://aistudio.google.com/app/apikey
   - BEA API: https://apps.bea.gov/API/signup/ (free registration)

   **M4 (Current Solutions):**
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   PORT=8000
   ```
   
   **Where to get this key:**
   - OpenAI API: https://platform.openai.com/api-keys

   **M5 (Idea Generation):**
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   PORT=8000
   ```
   
   **Where to get this key:**
   - Gemini API: https://aistudio.google.com/app/apikey

   **M6 (Market Analysis):**
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   NEWSAPI_KEY=your_newsapi_key_here (optional - for news fetching)
   PORT=8000
   ```
   
   **Where to get these keys:**
   - Gemini API: https://aistudio.google.com/app/apikey
   - NewsAPI: https://newsapi.org/register (free tier available)

3. **Deploy**: Railway will automatically detect `railway.json` and deploy

### Step 3: Get Service URLs

After deployment, each service will have a **unique Railway URL**:
- M2: `https://m2-problem-validation.up.railway.app`
- M3: `https://m3-problem-understanding.up.railway.app`
- M4: `https://m4-current-solutions.up.railway.app`
- M5: `https://m5-idea-generation.up.railway.app`
- M6: `https://m6-market-analysis.up.railway.app`

**Important:** Even though all modules use the same endpoint path (`/api/process`), they are **different services with different URLs**. The frontend will route to each module's specific URL.

Example:
- For M2: `POST https://m2-problem-validation.up.railway.app/api/process`
- For M3: `POST https://m3-problem-understanding.up.railway.app/api/process`

## Testing Each Module

### Using curl:

```bash
# M2
curl -X POST https://m2-problem-validation.up.railway.app/api/process \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How to improve remote work productivity"}'

# M3
curl -X POST https://m3-problem-understanding.up.railway.app/api/process \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Employee turnover in startups"}'

# Health check
curl https://m2-problem-validation.up.railway.app/health
```

## Module Responsibilities

Each module maintainer is responsible for:
1. **Deployment**: Deploying their module to Railway
2. **Environment Variables**: Setting required API keys
3. **Monitoring**: Checking health endpoints and logs
4. **Updates**: Pushing code changes (auto-deploys via Railway)

## Frontend Integration (Later)

When ready to integrate with frontend:
- Frontend will call each module's **unique Railway URL** + `/api/process` endpoint
- Example: `POST https://m2-module.up.railway.app/api/process`
- See `API_ROUTING.md` for detailed frontend integration guide
- Each module can be scaled independently
- Module failures are isolated (one module down doesn't affect others)

**Key Point:** Even though all modules use the same endpoint path (`/api/process`), they are **different services with different URLs**. The frontend routes to each module's specific Railway URL.

## Local Development

To test a module locally:

```bash
# For M2
cd modules/m2
uvicorn app:app --reload --port 8000

# Test
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test problem"}'
```

## Notes

- All modules share the same `requirements.txt` at the root
- Each module's `app.py` imports from the parent `modules/` directory
- Railway automatically handles Python dependencies via `requirements.txt`
- Services are independent and can be deployed/updated separately

