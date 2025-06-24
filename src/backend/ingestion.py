import asyncio
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from .schemas import IngestRequest
from .database import vita_db
from .file_processor import process_attachments
from .embedding import embedding_manager
from .permissions import permission_manager
from .utils import clean_text, redact_sensitive_info, split_text_for_embedding
from .progress_tracker import progress_tracker
from .ontology import ontology_manager
from .llm_client import llm_client
from .retry_utils import retry_on_api_error
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

def compute_content_hash(content: str, attachment_ids: List[str] = None) -> str:
    """
    Compute SHA-256 hash of content combined with sorted attachment IDs.
    
    Args:
        content: Cleaned text content
        attachment_ids: List of attachment IDs (SHA-256 hashes)
        
    Returns:
        SHA-256 hash string
    """
    # Combine content with sorted attachment IDs for consistent hashing
    hash_input = content
    if attachment_ids:
        sorted_attachments = sorted(attachment_ids)
        hash_input += "|attachments:" + ",".join(sorted_attachments)
    
    return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

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
        attachment_ids = []  # v6.1: Track attachment IDs for duplicate detection
        
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
                # v6.1: Get both text content and attachment IDs
                attachment_data = await process_attachments(req.attachments, return_ids=True)
                processed_attachments = 0
                
                if isinstance(attachment_data, dict):
                    # New format with attachment IDs
                    attachment_texts = attachment_data.get('texts', [])
                    attachment_ids = attachment_data.get('attachment_ids', [])
                else:
                    # Fallback to old format
                    attachment_texts = attachment_data
                    attachment_ids = []
                
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
        
        # v6.1: Compute content hash for duplicate detection
        content_hash = compute_content_hash(combined_content, attachment_ids)
        log_details.append(f"Computed content hash: {content_hash[:8]}...")
        
        # v6.1: Check for duplicate content
        if vita_db.check_content_duplicate(content_hash):
            log_details.append(f"DUPLICATE: Content hash {content_hash[:8]}... already exists")
            logger.info(f"Message {req.message_id}: Duplicate content detected (hash: {content_hash[:8]}...)")
            
            if batch_id:
                progress_tracker.add_log(
                    batch_id, req.message_id, "DUPLICATE", 
                    log_details, channel_name, f"Duplicate content hash: {content_hash[:8]}..."
                )
            
            # Mark as processed with hash but don't re-ingest
            try:
                vita_db.mark_message_processed_with_hash(req.message_id, content_hash, req.channel_id, req.user_id)
            except Exception as e:
                logger.warning(f"Failed to mark duplicate message {req.message_id}: {e}")
            
            return
        
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
                "total_chunks": len(chunks),
                # v6.1: Add content hash and attachment IDs for traceability
                "content_hash": content_hash,
                "attachment_ids": attachment_ids if attachment_ids else []
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
        
        # *** NEW: Enhanced Ontology Tagging and Knowledge Graph Update ***
        try:
            # Skip ontology enhancement if we're experiencing rate limits
            await _enhance_with_ontology_and_graph(req.message_id, combined_content, log_details)
        except Exception as ontology_error:
            error_str = str(ontology_error).lower()
            if "rate limit" in error_str or "429" in error_str:
                logger.warning(f"Skipping ontology enhancement due to rate limits for message {req.message_id}")
                log_details.append("Skipped ontology enhancement due to rate limits")
            else:
                logger.warning(f"Ontology enhancement failed for message {req.message_id}: {ontology_error}")
                log_details.append(f"Ontology enhancement failed: {str(ontology_error)}")
                # Don't fail the entire ingestion if ontology enhancement fails
        
        # Mark as processed with content hash
        vita_db.mark_message_processed_with_hash(req.message_id, content_hash, req.channel_id, req.user_id)
        log_details.append("Successfully marked as processed with content hash")
        
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

@retry_on_api_error(max_attempts=3, min_wait=1.0, max_wait=15.0)
async def _enhance_with_ontology_and_graph(message_id: str, content: str, log_details: List[str]):
    """
    Enhance message with ontology tagging and knowledge graph updates.
    
    v6.1: Enhanced with confidence scoring and gating for high-quality graph data.
    
    Args:
        message_id: Discord message ID
        content: Combined message content
        log_details: List to append progress details
    """
    try:
        logger.debug(f"Starting ontology enhancement for message {message_id}")
        
        # Step 1: Ontology Tagging with Confidence Scoring
        ontology_prompt = ontology_manager.create_ontology_prompt(content)
        if not ontology_prompt.strip():
            log_details.append("Skipped ontology tagging - no concepts defined")
            return
        
        # v6.1: Updated prompt to require confidence scores
        enhanced_prompt = ontology_prompt + """

IMPORTANT: For each concept you identify, you MUST include a confidence score (0.0 to 1.0) indicating how certain you are about:
1. The entity being correctly identified in the text
2. The entity being a legitimate business concept (not generic terms)
3. The entity belonging to the specified concept category

Response format (REQUIRED):
{
    "concepts": [
        {"concept": "Project", "entity": "Project Phoenix", "confidence": 0.95},
        {"concept": "Person", "entity": "John Doe", "confidence": 0.88}
    ]
}

Only include entities with confidence >= 0.5. Be conservative with scoring.
"""
        
        # Call LLM for ontology tagging
        tag_response = await llm_client.client.chat.completions.create(
            model=llm_client.chat_model,
            messages=[
                {"role": "system", "content": "You are an expert business analyst who identifies and extracts business concepts from text with confidence scoring. Always respond with valid JSON including confidence scores."},
                {"role": "user", "content": enhanced_prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        tag_response_text = tag_response.choices[0].message.content.strip()
        logger.debug(f"Ontology tagging response for {message_id}: {tag_response_text}")
        
        # Parse ontology tagging results
        try:
            tag_data = json.loads(tag_response_text)
            entities = tag_data.get('concepts', [])
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse ontology tagging JSON for message {message_id}")
            log_details.append("Ontology tagging failed - invalid JSON response")
            return
        
        if not entities:
            log_details.append("No ontology concepts identified")
            return
        
        # v6.1: Separate high-confidence entities for knowledge graph vs all entities for vector storage
        high_confidence_entities = []
        all_entities = []
        confidence_threshold = 0.75  # High threshold for knowledge graph
        
        for entity in entities:
            confidence = entity.get('confidence', 0.0)
            all_entities.append(entity)  # Store all entities in vector metadata
            
            if confidence >= confidence_threshold:
                high_confidence_entities.append(entity)
                logger.debug(f"High-confidence entity: {entity['concept']}:{entity['entity']} (conf: {confidence})")
            else:
                logger.debug(f"Low-confidence entity (vector only): {entity['concept']}:{entity['entity']} (conf: {confidence})")
        
        log_details.append(f"Identified {len(entities)} concepts total, {len(high_confidence_entities)} high-confidence for graph")
        
        # Step 2: Create/Update Knowledge Graph Nodes (only high-confidence)
        node_ids = {}
        for entity in high_confidence_entities:
            try:
                node_id = vita_db.create_or_get_node(
                    label=entity['concept'],
                    name=entity['entity'],
                    metadata={
                        'confidence': entity.get('confidence', 0.8),
                        'first_mentioned_message': message_id,
                        'source': 'ontology_extraction_v6.1',
                        'confidence_threshold': confidence_threshold
                    }
                )
                node_ids[entity['entity']] = node_id
                logger.debug(f"Created/updated high-confidence node for {entity['concept']}:{entity['entity']}")
            except Exception as e:
                logger.warning(f"Failed to create node for {entity['entity']}: {e}")
        
        # Step 3: Relationship Extraction with Confidence Scoring (only if we have multiple high-confidence entities)
        if len(high_confidence_entities) >= 2:
            relationship_prompt = ontology_manager.create_relationship_prompt(content, high_confidence_entities)
            if relationship_prompt.strip():
                try:
                    # v6.1: Enhanced relationship prompt with confidence requirements
                    enhanced_rel_prompt = relationship_prompt + """

IMPORTANT: For each relationship you identify, you MUST include a confidence score (0.0 to 1.0) indicating how certain you are about:
1. The relationship being explicitly or implicitly stated in the text
2. The relationship being meaningful and not coincidental
3. The relationship type being correctly identified

Response format (REQUIRED):
{
    "relationships": [
        {"source": "John Doe", "relationship": "manages", "target": "Project Phoenix", "confidence": 0.92}
    ]
}

Only include relationships with confidence >= 0.5. Be conservative with scoring.
"""
                    
                    # Call LLM for relationship extraction
                    rel_response = await llm_client.client.chat.completions.create(
                        model=llm_client.chat_model,
                        messages=[
                            {"role": "system", "content": "You are an expert business analyst who identifies relationships between business entities with confidence scoring. Always respond with valid JSON including confidence scores."},
                            {"role": "user", "content": enhanced_rel_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=800
                    )
                    
                    rel_response_text = rel_response.choices[0].message.content.strip()
                    logger.debug(f"Relationship extraction response for {message_id}: {rel_response_text}")
                    
                    # Parse relationship extraction results
                    try:
                        rel_data = json.loads(rel_response_text)
                        relationships = rel_data.get('relationships', [])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse relationship extraction JSON for message {message_id}")
                        relationships = []
                    
                    # Step 4: Create Graph Edges (only high-confidence relationships)
                    edges_created = 0
                    rel_confidence_threshold = 0.75  # High threshold for relationships
                    
                    for relationship in relationships:
                        confidence = relationship.get('confidence', 0.0)
                        if confidence >= rel_confidence_threshold:
                            source_name = relationship.get('source', '')
                            target_name = relationship.get('target', '')
                            rel_type = relationship.get('relationship', '')
                            
                            if source_name in node_ids and target_name in node_ids:
                                try:
                                    vita_db.create_edge(
                                        source_id=node_ids[source_name],
                                        target_id=node_ids[target_name],
                                        relationship=rel_type,
                                        metadata={
                                            'confidence': confidence,
                                            'source_message': message_id,
                                            'extraction_method': 'llm_analysis_v6.1',
                                            'confidence_threshold': rel_confidence_threshold
                                        },
                                        message_id=message_id
                                    )
                                    edges_created += 1
                                    logger.debug(f"Created high-confidence relationship: {source_name} -{rel_type}-> {target_name} (conf: {confidence})")
                                except Exception as e:
                                    logger.warning(f"Failed to create relationship edge: {e}")
                            else:
                                logger.debug(f"Skipping relationship - entities not in high-confidence graph: {source_name} -> {target_name}")
                        else:
                            logger.debug(f"Low-confidence relationship skipped: {relationship.get('source', '')} -> {relationship.get('target', '')} (conf: {confidence})")
                    
                    log_details.append(f"Created {edges_created} high-confidence relationship edges in knowledge graph")
                    
                except Exception as e:
                    logger.warning(f"Relationship extraction failed for message {message_id}: {e}")
                    log_details.append(f"Relationship extraction failed: {str(e)}")
        
        # v6.1: Store ALL entities (regardless of confidence) in vector metadata for discoverability
        # This will be handled in the metadata creation section of run_ingestion_task
        
        log_details.append("v6.1 Ontology enhancement completed with confidence gating")
        logger.info(f"Successfully enhanced message {message_id} with v6.1 ontology and graph data (confidence gating applied)")
        
    except Exception as e:
        logger.error(f"Ontology enhancement failed for message {message_id}: {e}")
        raise 