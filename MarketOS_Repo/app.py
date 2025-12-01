"""
FastAPI backend for StudioOS modules M2-M6
Standardized API contract for frontend integration
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import sys

# Import module functions
try:
    from modules.m2_module import run_m2
    from modules.m3_module import run_m3
    from modules.m4_module import run_m4
    from modules.m5_module import run_m5
    from modules.m6_module import run_m6
except ImportError as e:
    print(f"Warning: Module import error: {e}")
    # Fallback functions that return error
    def run_m2(prompt: str): return {"error": "M2 module not available"}
    def run_m3(prompt: str): return {"error": "M3 module not available"}
    def run_m4(prompt: str): return {"error": "M4 module not available"}
    def run_m5(prompt: str): return {"error": "M5 module not available"}
    def run_m6(prompt: str): return {"error": "M6 module not available"}

app = FastAPI(title="StudioOS API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Standardized request/response models
class ModuleRequest(BaseModel):
    prompt: str = Field(..., description="The problem space or prompt to process")

class ModuleResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: Optional[str] = None

class PipelineRequest(BaseModel):
    problemSpace: str = Field(..., description="Problem space description")
    enabledModules: Optional[List[str]] = Field(default=None, description="List of module IDs to run (M2-M6)")

class ModuleResult(BaseModel):
    id: str
    name: str
    status: str
    validatorScore: float
    retries: int
    tokens: int
    ms: int
    artifacts: List[Dict[str, Any]]
    startedAt: str
    completedAt: str

class PipelineResponse(BaseModel):
    runId: str
    mode: str
    problemSpace: str
    startedAt: str
    completedAt: str
    modules: List[ModuleResult]
    totalTokens: int
    totalMs: int

# Individual module endpoints
@app.post("/api/m2", response_model=ModuleResponse)
async def endpoint_m2(request: ModuleRequest):
    """Problem Validation module"""
    try:
        result = run_m2(request.prompt)
        return ModuleResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/m3", response_model=ModuleResponse)
async def endpoint_m3(request: ModuleRequest):
    """Problem Understanding & Cost Analysis module"""
    try:
        result = run_m3(request.prompt)
        return ModuleResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/m4", response_model=ModuleResponse)
async def endpoint_m4(request: ModuleRequest):
    """Current Solutions Analysis module"""
    try:
        result = run_m4(request.prompt)
        return ModuleResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/m5", response_model=ModuleResponse)
async def endpoint_m5(request: ModuleRequest):
    """Idea Generation module"""
    try:
        result = run_m5(request.prompt)
        return ModuleResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/m6", response_model=ModuleResponse)
async def endpoint_m6(request: ModuleRequest):
    """Market Analysis & Competitive Intelligence module"""
    try:
        result = run_m6(request.prompt)
        return ModuleResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pipeline endpoint (compatible with existing frontend)
@app.post("/pipeline", response_model=PipelineResponse)
async def pipeline_endpoint(request: PipelineRequest):
    """
    Main pipeline endpoint that runs modules in sequence.
    Compatible with existing frontend expectations.
    """
    import time
    from datetime import datetime
    
    run_id = f"run_{int(time.time() * 1000)}"
    problem_space = request.problemSpace
    
    # Determine which modules to run
    all_modules = ["M2", "M3", "M4", "M5", "M6"]
    enabled_modules = request.enabledModules if request.enabledModules else all_modules
    
    # Validate module IDs
    valid_modules = {"M2", "M3", "M4", "M5", "M6"}
    for module_id in enabled_modules:
        if module_id not in valid_modules:
            raise HTTPException(status_code=400, detail=f"Invalid module ID: {module_id}")
    
    results = []
    shared_state = {"problemSpace": problem_space}
    total_tokens = 0
    total_ms = 0
    
    # Module names mapping
    module_names = {
        "M2": "Problem Validation",
        "M3": "Problem Understanding & Cost Analysis",
        "M4": "Current Solutions Analysis",
        "M5": "Idea Generation",
        "M6": "Market Analysis & Competitive Intelligence"
    }
    
    try:
        for module_id in enabled_modules:
            started_at = datetime.utcnow().isoformat() + "Z"
            start_time = time.time()
            
            # Run the module
            if module_id == "M2":
                module_result = run_m2(problem_space)
            elif module_id == "M3":
                module_result = run_m3(problem_space)
            elif module_id == "M4":
                module_result = run_m4(problem_space)
            elif module_id == "M5":
                module_result = run_m5(problem_space)
            elif module_id == "M6":
                module_result = run_m6(problem_space)
            else:
                continue
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            completed_at = datetime.utcnow().isoformat() + "Z"
            
            # Store result in shared state for downstream modules
            shared_state[module_id] = module_result
            
            # Format as ModuleResult
            result = ModuleResult(
                id=module_id,
                name=module_names.get(module_id, module_id),
                status="succeeded",
                validatorScore=0.9,  # Default score, can be enhanced
                retries=0,
                tokens=0,  # Can be tracked if needed
                ms=elapsed_ms,
                artifacts=[{
                    "module": module_id,
                    "kind": "json",
                    "title": f"{module_id} Output",
                    "preview": str(module_result)[:120] if isinstance(module_result, str) else "Module completed",
                    "data": module_result
                }],
                startedAt=started_at,
                completedAt=completed_at
            )
            
            results.append(result)
            total_ms += elapsed_ms
        
        return PipelineResponse(
            runId=run_id,
            mode="modular" if request.enabledModules else "automatic",
            problemSpace=problem_space,
            startedAt=results[0].startedAt if results else datetime.utcnow().isoformat() + "Z",
            completedAt=results[-1].completedAt if results else datetime.utcnow().isoformat() + "Z",
            modules=results,
            totalTokens=total_tokens,
            totalMs=total_ms
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "StudioOS API"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

