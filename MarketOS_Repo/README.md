# marketOS-Backend

Backend services for StudioOS modules (M2-M6). Each module is deployed as an independent FastAPI microservice on Railway.

## Structure

```
marketOS-Backend/
├── modules/
│   ├── m2/              # Problem Validation service
│   │   ├── app.py       # FastAPI app
│   │   └── railway.json # Railway deployment config
│   ├── m3/              # Problem Understanding service
│   ├── m4/              # Current Solutions Analysis service
│   ├── m5/              # Idea Generation service
│   ├── m6/              # Market Analysis service
│   ├── m2_module.py     # M2 module logic
│   ├── m3_module.py     # M3 module logic
│   ├── m4_module.py     # M4 module logic
│   ├── m5_module.py     # M5 module logic
│   └── m6_module.py     # M6 module logic
├── requirements.txt     # Python dependencies
├── DEPLOYMENT.md        # Deployment guide
├── API_ROUTING.md       # Frontend integration guide
└── API_KEYS_SETUP.md   # API keys setup guide
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

See `API_KEYS_SETUP.md` for detailed instructions on obtaining and setting API keys.

### 3. Deploy to Railway

Each module is deployed independently:

1. Create a new Railway project for each module
2. Set environment variables in Railway dashboard
3. Railway auto-detects `railway.json` in each module folder

See `DEPLOYMENT.md` for step-by-step deployment instructions.

## API Endpoints

All modules follow the same standardized API contract:

- **Endpoint**: `POST /api/process`
- **Request**: `{"prompt": "your problem statement"}`
- **Response**: `{"success": true, "data": {...}, "message": null}`
- **Health**: `GET /health`

## Documentation

- **[DEPLOYMENT.md](./DEPLOYMENT.md)** - How to deploy each module to Railway
- **[API_ROUTING.md](./API_ROUTING.md)** - How frontend routes to each module
- **[API_KEYS_SETUP.md](./API_KEYS_SETUP.md)** - Step-by-step API key setup

## Module Responsibilities

Each module maintainer is responsible for:
- Deploying their module to Railway
- Setting required environment variables
- Monitoring service health
- Updating code (auto-deploys via Railway)

## Local Development

Test a module locally:

```bash
cd modules/m2
uvicorn app:app --reload --port 8000
```

## Notes

- Each module is a separate microservice with its own Railway URL
- All modules share the same `requirements.txt`
- Modules can be updated/deployed independently
- Frontend integration will be handled separately via Lovable

