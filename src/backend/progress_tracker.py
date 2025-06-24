import uuid
import time
from typing import Dict, List, Optional
from datetime import datetime
from .schemas import BatchProgress, IngestionLog
from .logger import get_logger

logger = get_logger(__name__)

class ProgressTracker:
    """Tracks progress of batch ingestion operations."""
    
    def __init__(self):
        self._active_batches: Dict[str, BatchProgress] = {}
        self._max_recent_logs = 10  # Keep last 10 log entries
    
    def create_batch(self, total_messages: int, channel_name: Optional[str] = None) -> str:
        """
        Create a new batch tracking entry.
        
        Args:
            total_messages: Total number of messages to process
            channel_name: Name of the channel being processed
            
        Returns:
            Unique batch ID
        """
        batch_id = str(uuid.uuid4())
        
        progress = BatchProgress(
            batch_id=batch_id,
            total_messages=total_messages,
            processed_count=0,
            success_count=0,
            error_count=0,
            skipped_count=0,
            current_channel=channel_name,
            recent_logs=[],
            status="PROCESSING"
        )
        
        self._active_batches[batch_id] = progress
        logger.info(f"Created batch {batch_id} for {total_messages} messages")
        
        return batch_id
    
    def add_log(self, batch_id: str, message_id: str, status: str, 
                details: List[str], channel_name: Optional[str] = None, 
                error_message: Optional[str] = None):
        """
        Add a log entry for a processed message.
        
        Args:
            batch_id: Batch identifier
            message_id: Discord message ID
            status: Processing status
            details: List of processing details
            channel_name: Channel name
            error_message: Error message if failed
        """
        if batch_id not in self._active_batches:
            logger.warning(f"Batch {batch_id} not found")
            return
        
        batch = self._active_batches[batch_id]
        
        # Create log entry
        log_entry = IngestionLog(
            message_id=message_id,
            status=status,
            details=details,
            timestamp=datetime.now().isoformat(),
            channel_name=channel_name,
            error_message=error_message
        )
        
        # Update counts
        batch.processed_count += 1
        if status == "SUCCESS":
            batch.success_count += 1
        elif status == "ERROR":
            batch.error_count += 1
        elif status == "SKIPPED":
            batch.skipped_count += 1
        
        # Add to recent logs (keep only last N entries)
        batch.recent_logs.append(log_entry)
        if len(batch.recent_logs) > self._max_recent_logs:
            batch.recent_logs = batch.recent_logs[-self._max_recent_logs:]
        
        # Update current channel if provided
        if channel_name:
            batch.current_channel = channel_name
        
        logger.debug(f"Added log for batch {batch_id}: {status} - {message_id}")
    
    def update_channel(self, batch_id: str, channel_name: str):
        """Update the current channel being processed."""
        if batch_id in self._active_batches:
            self._active_batches[batch_id].current_channel = channel_name
    
    def complete_batch(self, batch_id: str, final_status: str = "COMPLETED"):
        """
        Mark a batch as completed.
        
        Args:
            batch_id: Batch identifier
            final_status: Final status (COMPLETED or FAILED)
        """
        if batch_id in self._active_batches:
            self._active_batches[batch_id].status = final_status
            logger.info(f"Batch {batch_id} completed with status: {final_status}")
    
    def get_progress(self, batch_id: str) -> Optional[BatchProgress]:
        """Get current progress for a batch."""
        return self._active_batches.get(batch_id)
    
    def cleanup_old_batches(self, max_age_minutes: int = 60):
        """Clean up old completed batches."""
        current_time = time.time()
        to_remove = []
        
        for batch_id, batch in self._active_batches.items():
            if batch.status in ["COMPLETED", "FAILED"]:
                # Keep some recent ones but remove old ones
                if len(to_remove) < 10:  # Keep 10 most recent completed
                    continue
                to_remove.append(batch_id)
        
        # Always clean up batches older than max_age_minutes
        cutoff_time = current_time - (max_age_minutes * 60)
        for batch_id, batch in list(self._active_batches.items()):
            # Convert ISO string to timestamp for comparison
            try:
                if batch.recent_logs:
                    batch_time = datetime.fromisoformat(batch.recent_logs[0].timestamp).timestamp()
                if batch_time < cutoff_time:
                    to_remove.append(batch_id)
            except:
                # If we can't parse time, remove it to be safe
                to_remove.append(batch_id)
        
        # Remove duplicates and clean up
        to_remove = list(set(to_remove))
        for batch_id in to_remove:
            if batch_id in self._active_batches:
                del self._active_batches[batch_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old batches")

# Global progress tracker instance
progress_tracker = ProgressTracker() 