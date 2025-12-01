"""
Module M4: Current Solutions Analysis - Independent FastAPI Service
Deployable as standalone microservice
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Railway puts everything in app/, so use direct import
# Import lazily to avoid errors on startup if API keys aren't set
def get_run_m4():
    """Lazy import to avoid startup errors."""
    from m4_module import run_m4
    return run_m4

app = FastAPI(title="StudioOS M4 - Current Solutions Analysis", version="1.0.0")

# Thread pool executor for running blocking synchronous code
executor = ThreadPoolExecutor(max_workers=2)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModuleRequest(BaseModel):
    prompt: str = Field(..., description="The problem statement to analyze")
    user_role: Optional[str] = Field(None, description="User role: 'founder', 'investor', 'student', or 'studio_partner'")

class ModuleResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "StudioOS M4 - Current Solutions Analysis",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/api/process": {
                "method": "POST",
                "description": "Process a problem statement and return competitive analysis + Reddit VoC",
                "request_body": {
                    "prompt": "string - The problem statement to analyze",
                    "user_role": "string (optional) - User role: 'founder', 'investor', 'student', or 'studio_partner'"
                },
                "example": {
                    "prompt": "The current hiring process creates a destructive cycle where candidates feel compelled to game the system while companies struggle to see past the noise.",
                    "user_role": "founder"
                }
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint"
            },
            "/docs": {
                "method": "GET",
                "description": "Interactive API documentation (Swagger UI)"
            }
        },
        "response_structure": {
            "success": "boolean",
            "data": {
                "problem": "Normalized problem data",
                "landscape": "Solution landscape analysis",
                "competitive_analysis": {
                    "structured": "Parsed competitors array",
                    "markdown": "Full competitive analysis markdown"
                },
                "reddit_summary": "Reddit solutions summary",
                "reddit_data": "Array of Reddit posts/comments",
                "output_files": "Paths to saved files"
            }
        }
    }

@app.get("/api/process")
async def process_m4_get():
    """Inform users that POST is required for this endpoint"""
    raise HTTPException(
        status_code=405,
        detail={
            "error": "Method Not Allowed",
            "message": "This endpoint only accepts POST requests. Please use POST with a JSON body containing 'prompt' field.",
            "example": {
                "method": "POST",
                "url": "/api/process",
                "body": {
                    "prompt": "Your problem statement here",
                    "user_role": "founder"  # optional: 'founder', 'investor', 'student', or 'studio_partner'
                }
            }
        }
    )

@app.post("/api/process")
async def process_m4(request: ModuleRequest):
    """Process current solutions analysis request"""
    try:
        run_m4_func = get_run_m4()
        
        # Run the blocking synchronous function in a thread pool executor
        # This prevents blocking the async event loop and eliminates PRAW warnings
        # By running in a separate thread, PRAW won't detect the async context
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, run_m4_func, request.prompt, request.user_role)
        
        # Return properly formatted JSON response
        response_data = {
            "success": True,
            "data": result
        }
        
        # Use JSONResponse to ensure proper formatting
        return JSONResponse(
            content=response_data,
            media_type="application/json"
        )
    except Exception as e:
        import traceback
        error_detail = str(e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "module": "M4", "service": "Current Solutions Analysis"}

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup thread pool executor on shutdown"""
    executor.shutdown(wait=True)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

