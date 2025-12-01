# API Keys Setup Guide

This guide explains how to obtain and configure API keys for each module.

## Overview

Each module requires different API keys. You'll set these in Railway's environment variables for each service.

## Module M2 - Problem Validation

### Required Keys:
- **GEMINI_API_KEY** - Google Gemini API
- **REDDIT_CLIENT_ID** - Reddit API Client ID
- **REDDIT_CLIENT_SECRET** - Reddit API Client Secret
- **REDDIT_USER_AGENT** - Reddit User Agent string

### How to Get:

1. **Gemini API Key:**
   - Go to: https://aistudio.google.com/app/apikey
   - Sign in with Google account
   - Click "Create API Key"
   - Copy the key

2. **Reddit API Keys:**
   - Go to: https://www.reddit.com/prefs/apps
   - Scroll down and click "create another app..."
   - Fill in:
     - **Name**: StudioOS M2 (or any name)
     - **Type**: script
     - **Description**: (optional)
     - **About URL**: (leave blank)
     - **Redirect URI**: http://localhost:8080 (or any URL)
   - Click "create app"
   - You'll see:
     - **client_id**: The string under "personal use script" (14 characters)
     - **secret**: The "secret" field (27 characters)
   - **User Agent**: Format: `StudioOS-M2/1.0 (by u/yourusername)`

### Set in Railway:
```
GEMINI_API_KEY=AIzaSy...
REDDIT_CLIENT_ID=yr3ojj7Dfu0MW2FF9SWCrQ
REDDIT_CLIENT_SECRET=K8ZnT8yaU-idVU86Ec660HwMkUVBTw
REDDIT_USER_AGENT=StudioOS-M2/1.0 (by u/yourusername)
```

---

## Module M3 - Problem Understanding

### Required Keys:
- **GEMINI_API_KEY** - Google Gemini API
- **BEA_API_KEY** - Bureau of Economic Analysis API (optional)

### How to Get:

1. **Gemini API Key:**
   - Same as M2 above

2. **BEA API Key (Optional):**
   - Go to: https://apps.bea.gov/API/signup/
   - Fill in registration form
   - Verify email
   - You'll receive API key via email

### Set in Railway:
```
GEMINI_API_KEY=AIzaSy...
BEA_API_KEY=your_bea_key_here (optional)
```

---

## Module M4 - Current Solutions Analysis

### Required Keys:
- **OPENAI_API_KEY** - OpenAI API Key

### How to Get:

1. **OpenAI API Key:**
   - Go to: https://platform.openai.com/api-keys
   - Sign in or create account
   - Click "Create new secret key"
   - Name it (e.g., "StudioOS M4")
   - Copy the key immediately (you won't see it again)
   - **Note**: You'll need billing enabled (free tier available)

### Set in Railway:
```
OPENAI_API_KEY=sk-proj-...
```

---

## Module M5 - Idea Generation

### Required Keys:
- **GEMINI_API_KEY** - Google Gemini API

### How to Get:

1. **Gemini API Key:**
   - Same as M2 above

### Set in Railway:
```
GEMINI_API_KEY=AIzaSy...
```

---

## Module M6 - Market Analysis

### Required Keys:
- **GEMINI_API_KEY** - Google Gemini API
- **NEWSAPI_KEY** - NewsAPI Key (optional)

### How to Get:

1. **Gemini API Key:**
   - Same as M2 above

2. **NewsAPI Key (Optional):**
   - Go to: https://newsapi.org/register
   - Sign up for free account
   - Verify email
   - Get API key from dashboard
   - Free tier: 100 requests/day

### Set in Railway:
```
GEMINI_API_KEY=AIzaSy...
NEWSAPI_KEY=your_newsapi_key (optional)
```

---

## Setting Environment Variables in Railway

### Step-by-Step:

1. **Open Railway Dashboard:**
   - Go to your project
   - Click on the service (e.g., "M2 - Problem Validation")

2. **Go to Variables Tab:**
   - Click "Variables" in the left sidebar

3. **Add Variables:**
   - Click "New Variable"
   - Enter variable name (e.g., `GEMINI_API_KEY`)
   - Enter variable value (e.g., `AIzaSy...`)
   - Click "Add"

4. **Repeat for all required keys:**
   - Add each key as a separate variable
   - Don't include quotes around values

5. **Redeploy:**
   - Railway will automatically redeploy when you add variables
   - Or click "Redeploy" manually

### Example Railway Variables Tab:
```
Variable Name          | Value
----------------------|------------------
GEMINI_API_KEY        | AIzaSy...
REDDIT_CLIENT_ID      | yr3ojj7Dfu0MW2FF9SWCrQ
REDDIT_CLIENT_SECRET   | K8ZnT8yaU-idVU86Ec660HwMkUVBTw
REDDIT_USER_AGENT      | StudioOS-M2/1.0 (by u/yourusername)
```

---

## Security Best Practices

1. **Never commit API keys to Git:**
   - ✅ All keys are in `.gitignore`
   - ✅ Use Railway environment variables
   - ✅ Use `.env.local` for local development (not committed)

2. **Rotate keys regularly:**
   - Regenerate keys if exposed
   - Update in Railway immediately

3. **Use different keys per module:**
   - If you have multiple Gemini keys, use different ones per module
   - Helps with rate limiting and tracking

4. **Monitor usage:**
   - Check API dashboards regularly
   - Set up billing alerts

---

## Local Development

For local testing, create a `.env` file in the project root:

```bash
# .env (not committed to git)
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=StudioOS-M2/1.0 (by u/yourusername)
```

Python will automatically load these via `os.getenv()`.

---

## Troubleshooting

### "API key not found" error:
- Check variable name spelling in Railway
- Ensure no extra spaces in values
- Redeploy service after adding variables

### "Invalid API key" error:
- Verify key is correct (copy-paste again)
- Check if key has expired or been revoked
- Ensure billing is enabled (for OpenAI)

### Rate limiting:
- Check API usage in provider dashboard
- Consider using different keys per module
- Implement retry logic in code

---

## Quick Reference

| Module | Required Keys | Optional Keys |
|--------|--------------|---------------|
| M2 | GEMINI_API_KEY, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT | - |
| M3 | GEMINI_API_KEY | BEA_API_KEY |
| M4 | OPENAI_API_KEY | - |
| M5 | GEMINI_API_KEY | - |
| M6 | GEMINI_API_KEY | NEWSAPI_KEY |

