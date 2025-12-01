'''
curl -s -X POST http://localhost:8000/api/process \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"how to stay productive working remote","max_results":5,"include_report":true}' | jq -r '.data.report_text'

'''

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os
from m2 import run_m2

app = FastAPI(title="StudioOS M2 - Problem Validation", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class ModuleRequest(BaseModel):
    prompt: str = Field(..., description="The problem space or prompt to process")
    max_results: Optional[int] = Field(10, description="Max reddit threads to pull")
    include_report: Optional[bool] = Field(False, description="Include console-style report text")

class ModuleResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: Optional[str] = None

@app.post("/api/process", response_model=ModuleResponse)
async def process_m2(request: ModuleRequest):
    try:
        result = run_m2(
            prompt=request.prompt,
            max_results=request.max_results or 10,
            include_report=bool(request.include_report)
        )
        return ModuleResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "module": "M2", "service": "Problem Validation"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
