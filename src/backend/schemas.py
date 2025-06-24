from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class IngestRequest(BaseModel):
    """Request model for message ingestion."""
    message_id: str = Field(..., description="Discord message ID")
    channel_id: str = Field(..., description="Discord channel ID")
    user_id: str = Field(..., description="Discord user ID")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    attachments: Optional[List[str]] = Field(default=None, description="List of attachment URLs")
    thread_id: Optional[str] = Field(default=None, description="Discord thread ID if in thread")
    roles: Optional[List[str]] = Field(default=None, description="User roles for permission filtering")

class QueryRequest(BaseModel):
    """Request model for knowledge queries."""
    user_id: str = Field(..., description="Discord user ID")
    channel_id: str = Field(..., description="Discord channel ID")
    roles: List[str] = Field(..., description="User roles for permission filtering")
    question: str = Field(..., description="User's question")
    top_k: int = Field(default=5, description="Number of top results to retrieve")

class QueryResponse(BaseModel):
    """Response model for knowledge queries."""
    answer: str = Field(..., description="Generated answer")
    citations: List[Dict] = Field(..., description="List of source citations")
    confidence: float = Field(..., description="Confidence score of the answer")

class BatchIngestRequest(BaseModel):
    """Request model for batch message ingestion."""
    messages: List[IngestRequest] = Field(..., description="List of messages to ingest")

class IngestionLog(BaseModel):
    """Detailed log entry for a single message ingestion."""
    message_id: str
    status: str  # "SUCCESS", "SKIPPED", "ERROR", "PROCESSING"
    details: List[str] = []
    timestamp: str
    channel_name: Optional[str] = None
    error_message: Optional[str] = None

class BatchProgress(BaseModel):
    """Progress tracking for batch ingestion operations."""
    batch_id: str
    total_messages: int
    processed_count: int
    success_count: int
    error_count: int
    skipped_count: int
    current_channel: Optional[str] = None
    recent_logs: List[IngestionLog] = []
    status: str  # "PROCESSING", "COMPLETED", "FAILED"
    
class ProgressResponse(BaseModel):
    """Response for progress queries."""
    progress: BatchProgress
    message: str 