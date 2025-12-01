# Railway Deployment Troubleshooting for Module 4

## "Not Found" Error Fix

If you see "Not Found" error on Railway, follow these steps:

### Step 1: Check Railway Settings

1. **Root Directory**: 
   - Go to Railway Dashboard → Your Project → Settings
   - Under "Root Directory", set it to: `/modules/m4`
   - Click "Save"

2. **Environment Variables**:
   - Go to Variables tab
   - Make sure these are set:
     - `OPENAI_API_KEY` (REQUIRED)
     - `REDDIT_CLIENT_ID` (optional, but required for Reddit features)
     - `REDDIT_CLIENT_SECRET` (optional, but required for Reddit features)
     - `REDDIT_USER_AGENT` (optional, defaults to "MarketOS Module 4 by u/MarketOS")
     - `OPENAI_MODEL` (optional, defaults to "gpt-4o-mini")
     - `OUTPUT_DIR` (optional, defaults to "module4_outputs")

### Step 2: Check Deployment Logs

1. Go to Railway Dashboard → Your Project
2. Click on "Deployments" tab
3. Check the latest deployment logs for errors
4. Look for:
   - Import errors
   - Missing dependencies
   - API key errors
   - Build failures

### Step 3: Verify railway.json

Make sure `railway.json` in `/modules/m4/` contains:
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS",
    "buildCommand": "pip install -r requirements.txt"
  },
  "deploy": {
    "startCommand": "uvicorn app:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### Step 4: Enable Public Networking

1. Go to Settings → Networking
2. Click "Generate Domain" if not already done
3. This creates your public HTTPS URL

### Step 5: Test Health Endpoint

Once deployed, test:
```bash
curl https://your-railway-url.up.railway.app/health
```

Should return:
```json
{"status": "healthy", "module": "M4", "service": "Current Solutions Analysis"}
```

### Step 6: Common Issues

**Issue**: Service won't start
- **Fix**: Check that `OPENAI_API_KEY` is set in Railway variables

**Issue**: Import errors
- **Fix**: Make sure `requirements.txt` has all dependencies
- **Fix**: Check Railway logs for missing packages

**Issue**: Port binding errors
- **Fix**: Make sure start command uses `$PORT` variable
- **Fix**: Railway automatically sets PORT, don't hardcode it

**Issue**: Module not found
- **Fix**: Verify root directory is set to `/modules/m4`
- **Fix**: Make sure `app.py` and `m4_module.py` are in the same directory

### Step 7: Redeploy

After making changes:
1. Commit and push to GitHub:
   ```bash
   git add modules/m4/
   git commit -m "Fix Railway deployment"
   git push origin m4
   ```
2. Railway will auto-deploy from the branch
3. Wait for deployment to complete
4. Check logs for any errors

### Step 8: Verify Deployment

1. Health check: `GET /health` should return 200
2. Test API: `POST /api/process` with `{"prompt": "test problem"}`
3. Check logs for any runtime errors

## Testing Locally

Before deploying, test locally:
```bash
cd modules/m4
export OPENAI_API_KEY="your-key"
export REDDIT_CLIENT_ID="your-id"
export REDDIT_CLIENT_SECRET="your-secret"
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Then test:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test problem statement"}'
```


