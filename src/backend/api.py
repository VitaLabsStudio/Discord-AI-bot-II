import os
from typing import List
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Header
from fastapi.security import HTTPBearer
from dotenv import load_dotenv
from .schemas import IngestRequest, QueryRequest, QueryResponse, BatchIngestRequest, ProgressResponse
from .ingestion import run_ingestion_task, run_batch_ingestion_task
from .database import vita_db, is_processed
from .query_router import query_router
from .embedding import embedding_manager
from .permissions import permission_manager
from .llm_client import llm_client
from .progress_tracker import progress_tracker
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
        # Log the incoming request for verification
        logger.info(f"Received ingestion request for message {request.message_id}")
        
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
    Batch ingest multiple Discord messages with progress tracking.
    
    Args:
        request: BatchIngestRequest with list of messages
        background_tasks: FastAPI background tasks
        
    Returns:
        202 Accepted response with batch_id for tracking
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
        
        # Create batch tracker
        batch_id = progress_tracker.create_batch(len(ingest_requests))
        
        # Add batch ingestion task to background queue
        background_tasks.add_task(run_batch_ingestion_task, ingest_requests, batch_id)
        
        logger.info(f"Queued batch ingestion task for {len(ingest_requests)} messages with batch_id: {batch_id}")
        
        return {
            "status": "accepted",
            "batch_id": batch_id,
            "message_count": len(ingest_requests),
            "valid_messages": len(ingest_requests),
            "total_submitted": len(request.messages),
            "message": "Batch ingestion task queued for processing",
            "progress_endpoint": f"/progress/{batch_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to queue batch ingestion task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue batch ingestion task: {str(e)}")

@app.get("/progress/{batch_id}", response_model=ProgressResponse)
async def get_progress(
    batch_id: str,
    _: bool = Depends(verify_api_key)
):
    """
    Get real-time progress for a batch ingestion operation.
    
    Args:
        batch_id: Unique batch identifier
        
    Returns:
        Progress information and recent logs
    """
    try:
        progress = progress_tracker.get_progress(batch_id)
        
        if not progress:
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
        
        return ProgressResponse(
            progress=progress,
            message=f"Batch progress: {progress.processed_count}/{progress.total_messages} processed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get progress for batch {batch_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get progress: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_knowledge(
    request: QueryRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Query the knowledge base using enhanced RAG pipeline with query routing.
    
    Args:
        request: QueryRequest with user question and context
        
    Returns:
        QueryResponse with answer and citations
    """
    try:
        logger.info(f"Processing query from user {request.user_id}")
        
        # First, try to route the query to FAQ cache
        routed_response = query_router.route_query(request.question, request.user_id)
        if routed_response:
            # Return cached response immediately
            response = QueryResponse(
                answer=routed_response["answer"],
                citations=routed_response["citations"],
                confidence=routed_response["confidence"]
            )
            logger.info(f"Query answered by router ({routed_response['source']})")
            return response
        
        # If not routed, proceed with full AI pipeline
        logger.info(f"Query routed to AI pipeline for user {request.user_id}")
        
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

@app.post("/feedback")
async def record_feedback(
    request: dict,
    _: bool = Depends(verify_api_key)
):
    """
    Record user feedback on AI responses.
    
    Args:
        request: Dictionary with feedback data
        
    Returns:
        Success confirmation
    """
    try:
        required_fields = ["query_text", "answer_text", "is_helpful", "user_id"]
        missing_fields = [field for field in required_fields if field not in request]
        
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {missing_fields}"
            )
        
        feedback_id = vita_db.record_user_feedback(
            query_text=request["query_text"],
            answer_text=request["answer_text"],
            is_helpful=request["is_helpful"],
            user_id=request["user_id"],
            confidence_score=request.get("confidence_score")
        )
        
        return {
            "message": "Feedback recorded successfully",
            "feedback_id": feedback_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")

@app.delete("/delete_message/{message_id}")
async def delete_message_vectors(
    message_id: str,
    _: bool = Depends(verify_api_key)
):
    """
    Delete all vectors associated with a message ID from Pinecone.
    
    Args:
        message_id: Discord message ID
        
    Returns:
        Success confirmation
    """
    try:
        # Query for vectors with this message_id
        query_response = embedding_manager.index.query(
            vector=[0.0] * 1536,  # Dummy vector
            top_k=10000,  # Large number to get all matches
            include_metadata=True,
            filter={"message_id": message_id}
        )
        
        if not query_response.matches:
            return {
                "message": f"No vectors found for message {message_id}",
                "deleted_count": 0
            }
        
        # Extract vector IDs to delete
        vector_ids = [match.id for match in query_response.matches]
        
        # Delete vectors from Pinecone
        embedding_manager.index.delete(ids=vector_ids)
        
        logger.info(f"Deleted {len(vector_ids)} vectors for message {message_id}")
        
        return {
            "message": f"Successfully deleted vectors for message {message_id}",
            "deleted_count": len(vector_ids)
        }
        
    except Exception as e:
        logger.error(f"Failed to delete vectors for message {message_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete vectors: {str(e)}")

@app.get("/stats")
async def get_stats(_: bool = Depends(verify_api_key)):
    """
    Get enhanced system statistics including feedback data.
    
    Returns:
        System statistics
    """
    try:
        feedback_stats = vita_db.get_feedback_stats()
        faq_stats = query_router.get_stats()
        
        stats = {
            "processed_messages": vita_db.get_processed_count(),
            "feedback": feedback_stats,
            "faq_cache": faq_stats,
            "system_status": "operational"
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# Admin endpoints for Dead Letter Queue management
@app.get("/admin/dlq/stats")
async def get_dlq_stats(_: bool = Depends(verify_api_key)):
    """
    Get dead letter queue statistics (Admin only).
    
    Returns:
        DLQ statistics including failure types and counts
    """
    try:
        from .dead_letter_queue import enhanced_dlq
        
        stats = enhanced_dlq.get_stats()
        
        return {
            "stats": stats,
            "message": "DLQ statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get DLQ stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get DLQ stats: {str(e)}")

@app.get("/admin/dlq/items")
async def get_dlq_items(
    failure_type: str = None,
    limit: int = 10,
    _: bool = Depends(verify_api_key)
):
    """
    Get dead letter queue items (Admin only).
    
    Args:
        failure_type: Optional filter by failure type
        limit: Maximum number of items to return
        
    Returns:
        List of DLQ items
    """
    try:
        from .dead_letter_queue import enhanced_dlq, FailureType
        
        # Validate failure type if provided
        failure_type_enum = None
        if failure_type:
            try:
                failure_type_enum = FailureType(failure_type.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid failure_type. Valid types: {[ft.value for ft in FailureType]}"
                )
        
        # Validate limit
        if limit < 1 or limit > 100:
            limit = min(max(limit, 1), 100)
        
        items = enhanced_dlq.get_items(failure_type=failure_type_enum, limit=limit)
        
        # Convert items to dictionaries for JSON response
        items_data = [item.to_dict() for item in items]
        
        return {
            "items": items_data,
            "count": len(items_data),
            "message": f"Retrieved {len(items_data)} DLQ items"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get DLQ items: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get DLQ items: {str(e)}")

@app.post("/admin/dlq/cleanup")
async def cleanup_dlq(
    request: dict,
    _: bool = Depends(verify_api_key)
):
    """
    Clean up old dead letter queue items (Admin only).
    
    Args:
        request: Dictionary with 'days' parameter
        
    Returns:
        Number of items removed
    """
    try:
        from .dead_letter_queue import enhanced_dlq
        
        days = request.get("days", 30)
        
        # Validate days parameter
        if not isinstance(days, int) or days < 1 or days > 365:
            raise HTTPException(
                status_code=400,
                detail="Days parameter must be an integer between 1 and 365"
            )
        
        removed_count = enhanced_dlq.cleanup_old_items(days)
        
        return {
            "removed_count": removed_count,
            "message": f"Cleaned up {removed_count} items older than {days} days"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cleanup DLQ: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup DLQ: {str(e)}")

@app.delete("/admin/dlq/items/{item_id}")
async def remove_dlq_item(
    item_id: str,
    _: bool = Depends(verify_api_key)
):
    """
    Remove a specific dead letter queue item (Admin only).
    
    Args:
        item_id: ID of the item to remove
        
    Returns:
        Success message
    """
    try:
        from .dead_letter_queue import enhanced_dlq
        
        success = enhanced_dlq.remove_item(item_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"DLQ item {item_id} not found")
        
        return {
            "message": f"Successfully removed DLQ item {item_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove DLQ item {item_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove DLQ item: {str(e)}")

@app.post("/admin/dlq/items/{item_id}/retry")
async def retry_dlq_item(
    item_id: str,
    _: bool = Depends(verify_api_key)
):
    """
    Mark a dead letter queue item as retried (Admin only).
    
    Args:
        item_id: ID of the item to mark as retried
        
    Returns:
        Success message
    """
    try:
        from .dead_letter_queue import enhanced_dlq
        
        # Get the item first
        item = enhanced_dlq.get_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail=f"DLQ item {item_id} not found")
        
        # Mark as retried
        success = enhanced_dlq.mark_retry(item_id)
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to mark item {item_id} as retried")
        
        return {
            "message": f"Successfully marked DLQ item {item_id} as retried",
            "retry_count": item.retry_count + 1
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry DLQ item {item_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retry DLQ item: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", 8000))
    
    uvicorn.run(app, host=host, port=port, log_level="info") 