import os
from typing import List
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Header
from fastapi.security import HTTPBearer
from dotenv import load_dotenv
from .schemas import IngestRequest, QueryRequest, QueryResponse, BatchIngestRequest
from .ingestion import run_ingestion_task, run_batch_ingestion_task, is_processed
from .embedding import embedding_manager
from .permissions import permission_manager
from .llm_client import llm_client
from .logger import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VITA Discord AI Knowledge Assistant",
    description="Backend API for Discord AI knowledge management",
    version="1.0.0"
)

# Security
security = HTTPBearer()

async def verify_api_key(x_api_key: str = Header(...)):
    """
    Verify API key from request headers.
    
    Args:
        x_api_key: API key from X-API-Key header
        
    Returns:
        True if valid
        
    Raises:
        HTTPException: If API key is invalid
    """
    expected_key = os.getenv("BACKEND_API_KEY")
    if not expected_key:
        raise HTTPException(status_code=500, detail="Backend API key not configured")
    
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "VITA Discord AI Knowledge Assistant API", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "openai": "configured" if os.getenv("OPENAI_API_KEY") else "not_configured",
            "pinecone": "configured" if os.getenv("PINECONE_API_KEY") else "not_configured"
        }
    }

@app.post("/ingest", status_code=202)
async def ingest_message(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_api_key)
):
    """
    Ingest a Discord message for knowledge storage.
    
    CRITICAL FIX: Uses BackgroundTasks for async processing.
    Returns 202 Accepted immediately after queuing the task.
    
    Args:
        request: IngestRequest with message data
        background_tasks: FastAPI background tasks
        
    Returns:
        202 Accepted response
    """
    try:
        # Check for idempotency
        if is_processed(request.message_id):
            logger.info(f"Message {request.message_id} already processed")
            return {
                "status": "already_processed",
                "message_id": request.message_id
            }
        
        # Add ingestion task to background queue
        background_tasks.add_task(run_ingestion_task, request)
        
        logger.info(f"Queued ingestion task for message {request.message_id}")
        
        return {
            "status": "accepted",
            "message_id": request.message_id,
            "message": "Ingestion task queued for processing"
        }
        
    except Exception as e:
        logger.error(f"Failed to queue ingestion task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue ingestion task: {str(e)}")

@app.post("/batch_ingest", status_code=202)
async def batch_ingest_messages(
    request: BatchIngestRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_api_key)
):
    """
    Batch ingest multiple Discord messages.
    
    Args:
        request: BatchIngestRequest with list of messages
        background_tasks: FastAPI background tasks
        
    Returns:
        202 Accepted response
    """
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        if len(request.messages) > 1000:
            raise HTTPException(status_code=400, detail="Batch size too large (max 1000 messages)")
        
        # Convert to IngestRequest objects if needed with validation
        ingest_requests = []
        validation_errors = []
        
        for i, msg in enumerate(request.messages):
            try:
                if isinstance(msg, dict):
                    # Validate required fields before conversion
                    required_fields = ["message_id", "channel_id", "user_id", "content", "timestamp"]
                    missing_fields = [field for field in required_fields if field not in msg]
                    if missing_fields:
                        validation_errors.append(f"Message {i}: Missing fields {missing_fields}")
                        continue
                    
                    # Convert dict to IngestRequest
                    ingest_req = IngestRequest(**msg)
                else:
                    # Already an IngestRequest object
                    ingest_req = msg
                ingest_requests.append(ingest_req)
            except Exception as e:
                validation_errors.append(f"Message {i}: {str(e)}")
        
        if validation_errors:
            error_summary = f"Validation errors in batch: {'; '.join(validation_errors[:5])}"
            if len(validation_errors) > 5:
                error_summary += f" (and {len(validation_errors) - 5} more errors)"
            raise HTTPException(status_code=400, detail=error_summary)
        
        if not ingest_requests:
            raise HTTPException(status_code=400, detail="No valid messages to process")
        
        # Add batch ingestion task to background queue
        background_tasks.add_task(run_batch_ingestion_task, ingest_requests)
        
        logger.info(f"Queued batch ingestion task for {len(ingest_requests)} messages")
        
        return {
            "status": "accepted",
            "message_count": len(ingest_requests),
            "valid_messages": len(ingest_requests),
            "total_submitted": len(request.messages),
            "message": "Batch ingestion task queued for processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to queue batch ingestion task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue batch ingestion task: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_knowledge(
    request: QueryRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Query the knowledge base using RAG pipeline.
    
    Args:
        request: QueryRequest with user question and context
        
    Returns:
        QueryResponse with answer and citations
    """
    try:
        logger.info(f"Processing query from user {request.user_id}")
        
        # Create permission filter
        permission_filter = permission_manager.create_permission_filter(
            request.roles, 
            request.channel_id
        )
        
        # Search for relevant documents
        similar_docs = await embedding_manager.query_similar(
            request.question,
            top_k=request.top_k,
            filter_dict=permission_filter
        )
        
        # Generate answer using LLM
        llm_result = await llm_client.generate_answer(
            request.question,
            similar_docs,
            request.user_id
        )
        
        # Create citations from source documents
        citations = []
        for doc in similar_docs:
            metadata = doc.get("metadata", {})
            citation = {
                "message_id": metadata.get("message_id", ""),
                "channel_id": metadata.get("channel_id", ""),
                "user_id": metadata.get("user_id", ""),
                "timestamp": metadata.get("timestamp", ""),
                "content_preview": metadata.get("content", "")[:200] + "..." if len(metadata.get("content", "")) > 200 else metadata.get("content", ""),
                "score": doc.get("score", 0.0)
            }
            citations.append(citation)
        
        response = QueryResponse(
            answer=llm_result["answer"],
            citations=citations,
            confidence=llm_result["confidence"]
        )
        
        logger.info(f"Generated response with {len(citations)} citations")
        return response
        
    except Exception as e:
        logger.error(f"Failed to process query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

@app.get("/stats")
async def get_stats(_: bool = Depends(verify_api_key)):
    """
    Get system statistics.
    
    Returns:
        System statistics
    """
    try:
        from .processed_log import processed_log
        
        stats = {
            "processed_messages": processed_log.get_processed_count(),
            "system_status": "operational"
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", 8000))
    
    uvicorn.run(app, host=host, port=port, log_level="info") 