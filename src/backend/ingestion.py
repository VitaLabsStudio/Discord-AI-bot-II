import asyncio
from datetime import datetime
from typing import List, Dict, Any
from .schemas import IngestRequest
from .processed_log import processed_log
from .file_processor import process_attachments
from .embedding import embedding_manager
from .permissions import permission_manager
from .utils import clean_text, redact_sensitive_info, split_text_for_embedding
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
    return processed_log.is_processed(message_id)

def mark_processed(message_id: str):
    """
    Mark a message as processed.
    
    Args:
        message_id: Discord message ID
    """
    processed_log.mark_processed(message_id)

async def run_ingestion_task(req: IngestRequest):
    """
    Run the complete ingestion pipeline for a message.
    
    This function is designed to be executed as a background task.
    
    Args:
        req: IngestRequest with message data
    """
    try:
        logger.info(f"Starting ingestion for message {req.message_id}")
        
        # Check if already processed (double-check for race conditions)
        if is_processed(req.message_id):
            logger.info(f"Message {req.message_id} already processed, skipping")
            return
        
        # Collect all content
        all_content = []
        
        # Add message content if present
        if req.content and req.content.strip():
            cleaned_content = clean_text(req.content)
            redacted_content = redact_sensitive_info(cleaned_content)
            if redacted_content:
                all_content.append(redacted_content)
        
        # Process attachments if present
        if req.attachments:
            logger.info(f"Processing {len(req.attachments)} attachments for message {req.message_id}")
            try:
                attachment_texts = await process_attachments(req.attachments)
                for text in attachment_texts:
                    if text and text.strip():
                        cleaned_text_content = clean_text(text)
                        redacted_text_content = redact_sensitive_info(cleaned_text_content)
                        if redacted_text_content:
                            all_content.append(redacted_text_content)
            except Exception as e:
                logger.error(f"Failed to process attachments for message {req.message_id}: {e}")
        
        # Skip if no content to process
        if not all_content:
            logger.info(f"No content to process for message {req.message_id}")
            mark_processed(req.message_id)
            return
        
        # Combine all content
        combined_content = "\n\n".join(all_content)
        
        # Split content into chunks
        chunks = split_text_for_embedding(combined_content)
        if not chunks:
            logger.info(f"No chunks generated for message {req.message_id}")
            mark_processed(req.message_id)
            return
        
        logger.info(f"Generated {len(chunks)} chunks for message {req.message_id}")
        
        # Generate embeddings
        embeddings = await embedding_manager.embed_chunks(chunks)
        if not embeddings:
            logger.error(f"Failed to generate embeddings for message {req.message_id}")
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
        
        # Store embeddings in Pinecone
        embedding_manager.store_embeddings(embeddings, metadatas)
        
        # Mark as processed
        mark_processed(req.message_id)
        
        logger.info(f"Successfully completed ingestion for message {req.message_id}")
        
    except Exception as e:
        logger.error(f"Failed to ingest message {req.message_id}: {e}")
        # Don't mark as processed if there was an error
        raise

async def run_batch_ingestion_task(requests: List[IngestRequest]):
    """
    Run batch ingestion for multiple messages.
    
    Args:
        requests: List of IngestRequest objects
    """
    try:
        logger.info(f"Starting batch ingestion for {len(requests)} messages")
        
        # Filter out already processed messages
        pending_requests = [req for req in requests if not is_processed(req.message_id)]
        
        if not pending_requests:
            logger.info("All messages in batch already processed")
            return
        
        logger.info(f"Processing {len(pending_requests)} pending messages")
        
        # Process messages in parallel (but limit concurrency to avoid rate limits)
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent ingestions
        
        async def process_with_semaphore(req):
            async with semaphore:
                await run_ingestion_task(req)
        
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
        
    except Exception as e:
        logger.error(f"Failed to run batch ingestion: {e}")
        raise 