"""
Module M3: Problem Understanding & Cost Analysis - Independent FastAPI Service
Deployable as standalone microservice
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import os
import sys
import logging
import json

# Configure logging for Railway monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Railway puts everything in app/, so use direct import
from m3_module import run_m3

app = FastAPI(title="StudioOS M3 - Problem Understanding", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")

class ModuleRequest(BaseModel):
    prompt: str = Field(..., description="The problem space or prompt to process")
    user_context: Optional[str] = Field(None, description="User context from localStorage (studioos_user_context)")
    conversation_history: Optional[List[Message]] = Field(None, description="Previous conversation messages for context")

class ModuleResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: str = None

@app.post("/api/process", response_model=ModuleResponse)
async def process_m3(request: ModuleRequest):
    """Process problem understanding & cost analysis request"""
    try:
        # Log input
        logger.info(f"M3 Input - Prompt: {request.prompt}")
        if request.user_context:
            logger.info(f"M3 Input - User Context: {request.user_context}")
        if request.conversation_history:
            logger.info(f"M3 Input - Conversation History: {len(request.conversation_history)} messages")
        
        # Convert conversation history format if provided
        conversation_history = None
        if request.conversation_history:
            conversation_history = []
            for msg in request.conversation_history:
                conversation_history.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        result = run_m3(
            problem=request.prompt,
            user_context=request.user_context,
            conversation_history=conversation_history
        )
        
        # Log output and status
        logger.info(f"M3 Output - Status: Success")
        logger.info(f"M3 Output - Response: {json.dumps(result, indent=2)}")
        
        return ModuleResponse(success=True, data=result)
    except Exception as e:
        logger.error(f"M3 Output - Status: Error")
        logger.error(f"M3 Output - Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "module": "M3", "service": "Problem Understanding & Cost Analysis"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
