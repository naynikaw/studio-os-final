# StudioOS FastAPI Backend

This FastAPI backend provides a standardized API for all StudioOS modules (M2-M6).

## Architecture

**Monolithic FastAPI Service** - All modules run in a single service for:
- Simpler deployment
- Lower cost
- Faster inter-module communication
- Easier state management

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file or set these environment variables:

```bash
# API Keys
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key

# Reddit API (for M2)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_user_agent

# Optional: Override defaults
GEMINI_MODEL=gemini-1.5-pro
PORT=8000
```

### Run Locally

```bash
# Development
uvicorn app:app --reload --port 8000

# Production
uvicorn app:app --host 0.0.0.0 --port $PORT
```

## API Endpoints

### Individual Module Endpoints

Each module has its own endpoint:

- `POST /api/m2` - Problem Validation
- `POST /api/m3` - Problem Understanding & Cost Analysis
- `POST /api/m4` - Current Solutions Analysis
- `POST /api/m5` - Idea Generation
- `POST /api/m6` - Market Analysis & Competitive Intelligence

**Request:**
```json
{
  "prompt": "Your problem statement here"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    // Module-specific output
  },
  "message": null
}
```

### Pipeline Endpoint

Main endpoint for running multiple modules in sequence (compatible with frontend):

- `POST /pipeline`

**Request:**
```json
{
  "problemSpace": "Your problem statement",
  "enabledModules": ["M2", "M3", "M4", "M5", "M6"]  // Optional, defaults to all
}
```

**Response:**
```json
{
  "runId": "run_1234567890",
  "mode": "automatic",
  "problemSpace": "Your problem statement",
  "startedAt": "2024-01-01T00:00:00Z",
  "completedAt": "2024-01-01T00:05:00Z",
  "modules": [
    {
      "id": "M2",
      "name": "Problem Validation",
      "status": "succeeded",
      "validatorScore": 0.9,
      "artifacts": [...]
    }
  ],
  "totalTokens": 0,
  "totalMs": 5000
}
```

### Health Check

- `GET /health` - Returns service health status

## Deployment on Railway

1. Connect your GitHub repository to Railway
2. Railway will automatically detect the `railway.json` configuration
3. Set environment variables in Railway dashboard
4. Deploy!

The `railway.json` file configures:
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## Module Structure

Each module is in `modules/` directory:
- `m2_module.py` - Problem Validation (Reddit scraping + Gemini analysis)
- `m3_module.py` - Problem Understanding (arXiv + BEA + Gemini)
- `m4_module.py` - Competitive Analysis (OpenAI GPT-4)
- `m5_module.py` - Idea Generation (Gemini)
- `m6_module.py` - Market Analysis (Gemini + NewsAPI + Semantic Scholar)

## Notes

- All modules accept a `prompt` string as input
- All modules return JSON-serializable dictionaries
- Modules can share state through the pipeline endpoint
- Error handling returns structured error responses

