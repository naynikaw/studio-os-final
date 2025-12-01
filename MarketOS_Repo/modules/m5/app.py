"""
Module M5: Idea Generation - Independent FastAPI Service
Deployable as standalone microservice
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any
import os
import sys

# Add parent directory to import m5_module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from m5_module import run_m5

app = FastAPI(title="StudioOS M5 - Idea Generation", version="1.0.0")

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
async def process_m5(request: ModuleRequest):
    """Process idea generation request"""
    try:
        result = run_m5(request.prompt)
        return ModuleResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "module": "M5", "service": "Idea Generation"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

