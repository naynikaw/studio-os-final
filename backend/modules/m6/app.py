"""
Module M6: Market Analysis & Competitive Intelligence - Independent FastAPI Service
Deployable as standalone microservice
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any
import os
import sys

# Add parent directory to import m6_module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from m6_module import run_m6
except ModuleNotFoundError:  # pragma: no cover - runtime convenience
    from modules.m6.m6_module import run_m6

app = FastAPI(title="StudioOS M6 - Market Analysis", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModuleRequest(BaseModel):
    prompt: str = Field(..., description="The problem space or prompt to process")

class ModuleResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: str = None

@app.post("/api/process", response_model=ModuleResponse)
async def process_m6(request: ModuleRequest):
    """Process market analysis request"""
    try:
        result = run_m6(request.prompt)
        return ModuleResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "module": "M6", "service": "Market Analysis & Competitive Intelligence"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

