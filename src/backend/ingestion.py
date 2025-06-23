import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from .schemas import IngestRequest
from .database import vita_db
from .file_processor import process_attachments
from .embedding import embedding_manager
from .permissions import permission_manager
from .utils import clean_text, redact_sensitive_info, split_text_for_embedding
from .progress_tracker import progress_tracker
from .logger import get_logger

logger = get_logger(__name__)

def is_processed(message_id: str) -> bool:
    """
    Check if a message has been processed.
    
    Args:
        message_id: Discord message ID
        
    Returns:
        True if message has been processed
    """
    return vita_db.is_message_processed(message_id)

def mark_processed(message_id: str, channel_id: str = None, user_id: str = None):
    """
    Mark a message as processed.
    
    Args:
        message_id: Discord message ID
        channel_id: Optional channel ID
        user_id: Optional user ID
    """
    vita_db.mark_message_processed(message_id, channel_id, user_id)

async def run_ingestion_task(req: IngestRequest, batch_id: Optional[str] = None, channel_name: Optional[str] = None):
    """
    Run the complete ingestion pipeline for a message.
    
    This function is designed to be executed as a background task.
    
    Args:
        req: IngestRequest with message data
        batch_id: Optional batch ID for progress tracking
        channel_name: Optional channel name for progress tracking
    """
    log_details = []
    
    try:
        log_details.append(f"Starting ingestion for message {req.message_id}")
        logger.info(f"Starting ingestion for message {req.message_id}")
        
        # Check if already processed (double-check for race conditions)
        if is_processed(req.message_id):
            log_details.append("Message already processed, skipping")
            logger.info(f"Message {req.message_id} already processed, skipping")
            
            if batch_id:
                progress_tracker.add_log(
                    batch_id, req.message_id, "SKIPPED", 
                    log_details, channel_name
                )
            return
        
        # Collect all content
        all_content = []
        
        # Add message content if present
        if req.content and req.content.strip():
            cleaned_content = clean_text(req.content)
            redacted_content = redact_sensitive_info(cleaned_content)
            if redacted_content:
                all_content.append(redacted_content)
                log_details.append("Processed message text content")
        
        # Process attachments if present
        if req.attachments:
            log_details.append(f"Processing {len(req.attachments)} attachments")
            logger.info(f"Processing {len(req.attachments)} attachments for message {req.message_id}")
            try:
                attachment_texts = await process_attachments(req.attachments)
                processed_attachments = 0
                for text in attachment_texts:
                    if text and text.strip():
                        cleaned_text_content = clean_text(text)
                        redacted_text_content = redact_sensitive_info(cleaned_text_content)
                        if redacted_text_content:
                            all_content.append(redacted_text_content)
                            processed_attachments += 1
                log_details.append(f"Successfully processed {processed_attachments} attachments")
            except Exception as e:
                error_msg = f"Failed to process attachments: {e}"
                log_details.append(f"ERROR: {error_msg}")
                logger.error(f"Failed to process attachments for message {req.message_id}: {e}")
        
        # Skip if no content to process
        if not all_content:
            log_details.append("No content to process, skipping")
            logger.info(f"No content to process for message {req.message_id}")
            
            if batch_id:
                progress_tracker.add_log(
                    batch_id, req.message_id, "SKIPPED", 
                    log_details, channel_name
                )
            
            mark_processed(req.message_id, req.channel_id, req.user_id)
            return
        
        # Combine all content
        combined_content = "\n\n".join(all_content)
        log_details.append(f"Combined content length: {len(combined_content)} characters")
        logger.info(f"Message {req.message_id}: Combined and cleaned content length: {len(combined_content)}")
        
        # Split content into chunks
        chunks = split_text_for_embedding(combined_content)
        if not chunks:
            log_details.append("No chunks generated after splitting")
            logger.info(f"No chunks generated for message {req.message_id}")
            
            if batch_id:
                progress_tracker.add_log(
                    batch_id, req.message_id, "SKIPPED", 
                    log_details, channel_name
                )
            
            mark_processed(req.message_id, req.channel_id, req.user_id)
            return
        
        log_details.append(f"Split into {len(chunks)} chunks for embedding")
        logger.info(f"Message {req.message_id}: Split into {len(chunks)} chunks for embedding.")
        
        # Generate embeddings
        try:
            embeddings = await embedding_manager.embed_chunks(chunks)
            if not embeddings:
                error_msg = "Failed to generate embeddings"
                log_details.append(f"ERROR: {error_msg}")
                logger.error(f"Failed to generate embeddings for message {req.message_id}")
                
                if batch_id:
                    progress_tracker.add_log(
                        batch_id, req.message_id, "ERROR", 
                        log_details, channel_name, error_msg
                    )
                return
                
            log_details.append(f"Generated {len(embeddings)} embedding vectors")
            logger.info(f"Message {req.message_id}: Generated {len(embeddings)} embedding vectors.")
        except Exception as e:
            error_msg = f"Error generating embeddings: {e}"
            log_details.append(f"ERROR: {error_msg}")
            logger.error(f"Error generating embeddings for message {req.message_id}: {e}")
            
            if batch_id:
                progress_tracker.add_log(
                    batch_id, req.message_id, "ERROR", 
                    log_details, channel_name, error_msg
                )
            return
        
        # Create metadata for each chunk
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "message_id": req.message_id,
                "channel_id": req.channel_id,
                "user_id": req.user_id,
                "content": chunk,
                "timestamp": req.timestamp.isoformat() if isinstance(req.timestamp, datetime) else str(req.timestamp),
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            
            # Add optional fields
            if req.thread_id:
                metadata["thread_id"] = req.thread_id
            
            if req.attachments:
                metadata["has_attachments"] = True
                metadata["attachment_count"] = len(req.attachments)
            
            # Add permission metadata
            if req.roles:
                metadata = permission_manager.add_permission_metadata(metadata, req.roles)
            
            metadatas.append(metadata)
            # Debug log for metadata creation (limit content preview)
            logger.debug(f"Message {req.message_id}, Chunk {i}: Metadata created, content preview: '{chunk[:80]}...'")
        
        log_details.append(f"Created metadata for {len(metadatas)} chunks")
        
        # Store embeddings in Pinecone
        try:
            embedding_manager.store_embeddings(embeddings, metadatas)
            log_details.append(f"Stored {len(chunks)} vectors in Pinecone")
            logger.info(f"Message {req.message_id}: Sent {len(chunks)} vectors to Pinecone for storage.")
        except Exception as e:
            error_msg = f"Error storing embeddings: {e}"
            log_details.append(f"ERROR: {error_msg}")
            logger.error(f"Error storing embeddings for message {req.message_id}: {e}")
            
            if batch_id:
                progress_tracker.add_log(
                    batch_id, req.message_id, "ERROR", 
                    log_details, channel_name, error_msg
                )
            return
        
        # Mark as processed
        mark_processed(req.message_id, req.channel_id, req.user_id)
        log_details.append("Successfully marked as processed")
        
        # Update progress tracker
        if batch_id:
            progress_tracker.add_log(
                batch_id, req.message_id, "SUCCESS", 
                log_details, channel_name
            )
        
        logger.info(f"Successfully completed ingestion for message {req.message_id}")
        
    except Exception as e:
        error_msg = f"Failed to ingest message {req.message_id}: {e}"
        log_details.append(f"CRITICAL ERROR: {e}")
        logger.error(error_msg)
        
        if batch_id:
            progress_tracker.add_log(
                batch_id, req.message_id, "ERROR", 
                log_details, channel_name, error_msg
            )
        
        # Don't mark as processed if there was an error
        raise

async def run_batch_ingestion_task(requests: List[IngestRequest], batch_id: Optional[str] = None):
    """
    Run batch ingestion for multiple messages with progress tracking.
    
    Args:
        requests: List of IngestRequest objects
        batch_id: Optional batch ID for progress tracking
    """
    try:
        logger.info(f"Starting batch ingestion for {len(requests)} messages")
        
        # Create batch tracker if not provided
        if not batch_id:
            batch_id = progress_tracker.create_batch(len(requests))
        
        # Filter out already processed messages
        pending_requests = [req for req in requests if not is_processed(req.message_id)]
        
        if not pending_requests:
            logger.info("All messages in batch already processed")
            progress_tracker.complete_batch(batch_id, "COMPLETED")
            return
        
        logger.info(f"Processing {len(pending_requests)} pending messages")
        
        # Process messages in parallel (but limit concurrency to avoid rate limits)
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent ingestions
        
        async def process_with_semaphore(req):
            async with semaphore:
                await run_ingestion_task(req, batch_id)
        
        # Execute all ingestion tasks
        tasks = [process_with_semaphore(req) for req in pending_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log results
        success_count = sum(1 for result in results if not isinstance(result, Exception))
        error_count = len(results) - success_count
        
        logger.info(f"Batch ingestion completed: {success_count} successful, {error_count} failed")
        
        if error_count > 0:
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch ingestion error for message {pending_requests[i].message_id}: {result}")
        
        # Mark batch as completed
        progress_tracker.complete_batch(batch_id, "COMPLETED")
        
    except Exception as e:
        logger.error(f"Failed to run batch ingestion: {e}")
        if batch_id:
            progress_tracker.complete_batch(batch_id, "FAILED")
        raise 