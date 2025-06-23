"""
Enhanced Dead Letter Queue system for failed ingestion items.
Provides better management, categorization, and retry capabilities.
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from filelock import FileLock
from .logger import get_logger

logger = get_logger(__name__)


class FailureType(Enum):
    """Categories of failures."""
    DOWNLOAD_ERROR = "download"
    PARSING_ERROR = "parsing"
    OCR_ERROR = "ocr"
    API_ERROR = "api"
    NETWORK_ERROR = "network"
    VALIDATION_ERROR = "validation"
    UNKNOWN_ERROR = "unknown"


@dataclass
class DeadLetterItem:
    """Enhanced dead letter queue item with more metadata."""
    id: str
    url: str
    error: str
    failure_type: FailureType
    step: str
    timestamp: datetime
    message_id: Optional[str] = None
    channel_id: Optional[str] = None
    retry_count: int = 0
    last_retry: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['failure_type'] = self.failure_type.value
        if self.last_retry:
            result['last_retry'] = self.last_retry.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeadLetterItem':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('last_retry'):
            data['last_retry'] = datetime.fromisoformat(data['last_retry'])
        data['failure_type'] = FailureType(data['failure_type'])
        return cls(**data)


class EnhancedDeadLetterQueue:
    """Enhanced dead letter queue with management capabilities."""
    
    def __init__(self, queue_file: str = "dead_letter_queue.json"):
        self.queue_file = queue_file
        self.lock_file = f"{queue_file}.lock"
        self._file_lock = FileLock(self.lock_file)
        self._items: List[DeadLetterItem] = []
        self._load_items()
    
    def _load_items(self):
        """Load items from file."""
        try:
            with self._file_lock:
                if os.path.exists(self.queue_file):
                    with open(self.queue_file, 'r') as f:
                        data = json.load(f)
                        self._items = [DeadLetterItem.from_dict(item) for item in data.get('items', [])]
                        logger.info(f"Loaded {len(self._items)} dead letter queue items")
        except Exception as e:
            logger.error(f"Failed to load dead letter queue: {e}")
            self._items = []
    
    def _save_items(self):
        """Save items to file."""
        try:
            data = {
                'items': [item.to_dict() for item in self._items],
                'last_updated': datetime.now().isoformat()
            }
            
            temp_file = self.queue_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            os.rename(temp_file, self.queue_file)
            logger.debug(f"Saved {len(self._items)} dead letter queue items")
        except Exception as e:
            logger.error(f"Failed to save dead letter queue: {e}")
    
    def add_item(self, url: str, error: str, failure_type: FailureType, step: str, 
                 message_id: Optional[str] = None, channel_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a failed item to the queue."""
        import uuid
        
        item = DeadLetterItem(
            id=str(uuid.uuid4()),
            url=url,
            error=error,
            failure_type=failure_type,
            step=step,
            timestamp=datetime.now(),
            message_id=message_id,
            channel_id=channel_id,
            metadata=metadata or {}
        )
        
        with self._file_lock:
            self._items.append(item)
            self._save_items()
            logger.warning(f"Added item to DLQ: {failure_type.value} error in {step}")
            
        return item.id
    
    def get_items(self, failure_type: Optional[FailureType] = None, 
                  limit: Optional[int] = None) -> List[DeadLetterItem]:
        """Get items from the queue, optionally filtered."""
        with self._file_lock:
            self._load_items()  # Reload to get latest
            
            items = self._items
            if failure_type:
                items = [item for item in items if item.failure_type == failure_type]
            
            # Sort by timestamp (newest first)
            items = sorted(items, key=lambda x: x.timestamp, reverse=True)
            
            if limit:
                items = items[:limit]
                
            return items
    
    def get_item(self, item_id: str) -> Optional[DeadLetterItem]:
        """Get a specific item by ID."""
        with self._file_lock:
            self._load_items()
            for item in self._items:
                if item.id == item_id:
                    return item
            return None
    
    def remove_item(self, item_id: str) -> bool:
        """Remove an item from the queue."""
        with self._file_lock:
            self._load_items()
            for i, item in enumerate(self._items):
                if item.id == item_id:
                    del self._items[i]
                    self._save_items()
                    logger.info(f"Removed item {item_id} from DLQ")
                    return True
            return False
    
    def mark_retry(self, item_id: str) -> bool:
        """Mark an item as retried."""
        with self._file_lock:
            self._load_items()
            for item in self._items:
                if item.id == item_id:
                    item.retry_count += 1
                    item.last_retry = datetime.now()
                    self._save_items()
                    logger.info(f"Marked item {item_id} as retried (count: {item.retry_count})")
                    return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the dead letter queue."""
        with self._file_lock:
            self._load_items()
            
            total_items = len(self._items)
            if total_items == 0:
                return {"total_items": 0}
            
            # Count by failure type
            failure_counts = {}
            retry_counts = {"no_retries": 0, "has_retries": 0}
            recent_failures = 0
            
            now = datetime.now()
            for item in self._items:
                # Count by failure type
                failure_type = item.failure_type.value
                failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
                
                # Count retries
                if item.retry_count > 0:
                    retry_counts["has_retries"] += 1
                else:
                    retry_counts["no_retries"] += 1
                
                # Recent failures (last 24 hours)
                if (now - item.timestamp) < timedelta(hours=24):
                    recent_failures += 1
            
            return {
                "total_items": total_items,
                "failure_types": failure_counts,
                "retry_stats": retry_counts,
                "recent_failures_24h": recent_failures,
                "oldest_failure": min(item.timestamp for item in self._items).isoformat(),
                "newest_failure": max(item.timestamp for item in self._items).isoformat()
            }
    
    def cleanup_old_items(self, days: int = 30) -> int:
        """Remove items older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._file_lock:
            self._load_items()
            original_count = len(self._items)
            self._items = [item for item in self._items if item.timestamp > cutoff]
            removed_count = original_count - len(self._items)
            
            if removed_count > 0:
                self._save_items()
                logger.info(f"Cleaned up {removed_count} old DLQ items")
            
            return removed_count


# Global enhanced dead letter queue instance
enhanced_dlq = EnhancedDeadLetterQueue() 