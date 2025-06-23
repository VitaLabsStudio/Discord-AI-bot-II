import json
import os
from typing import Set
from filelock import FileLock
from .logger import get_logger

logger = get_logger(__name__)

class ProcessedMessageLog:
    """Manages the log of processed messages for idempotency."""
    
    def __init__(self, log_file: str = "processed_messages.json"):
        self.log_file = log_file
        self.lock_file = f"{log_file}.lock"
        self._processed_messages: Set[str] = set()
        self._file_lock = FileLock(self.lock_file)
        self._load_processed_messages()
    
    def _load_processed_messages(self):
        """Load processed messages from file."""
        try:
            with self._file_lock:
                if os.path.exists(self.log_file):
                    with open(self.log_file, 'r') as f:
                        data = json.load(f)
                        self._processed_messages = set(data.get('processed_messages', []))
                        logger.info(f"Loaded {len(self._processed_messages)} processed messages")
                else:
                    logger.info("No existing processed messages log found, starting fresh")
                    
        except Exception as e:
            logger.error(f"Failed to load processed messages: {e}")
            self._processed_messages = set()
    
    def _save_processed_messages(self):
        """Save processed messages to file."""
        try:
            data = {
                'processed_messages': list(self._processed_messages)
            }
            
            # Write to temporary file first, then rename for atomicity
            temp_file = self.log_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            os.rename(temp_file, self.log_file)
            logger.debug(f"Saved {len(self._processed_messages)} processed messages")
            
        except Exception as e:
            logger.error(f"Failed to save processed messages: {e}")
    
    def is_processed(self, message_id: str) -> bool:
        """
        Check if a message has been processed.
        
        Args:
            message_id: Discord message ID
            
        Returns:
            True if message has been processed
        """
        with self._file_lock:
            # Reload from file to ensure we have the latest state across processes
            self._load_processed_messages_unsafe()
            return message_id in self._processed_messages
    
    def mark_processed(self, message_id: str):
        """
        Mark a message as processed.
        
        Args:
            message_id: Discord message ID to mark as processed
        """
        with self._file_lock:
            # Reload from file to ensure we have the latest state across processes
            self._load_processed_messages_unsafe()
            if message_id not in self._processed_messages:
                self._processed_messages.add(message_id)
                self._save_processed_messages()
                logger.debug(f"Marked message {message_id} as processed")
    
    def _load_processed_messages_unsafe(self):
        """Load processed messages from file without acquiring lock (internal use only)."""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self._processed_messages = set(data.get('processed_messages', []))
        except Exception as e:
            logger.error(f"Failed to reload processed messages: {e}")
            self._processed_messages = set()
    
    def get_processed_count(self) -> int:
        """
        Get the number of processed messages.
        
        Returns:
            Number of processed messages
        """
        with self._file_lock:
            self._load_processed_messages_unsafe()
            return len(self._processed_messages)
    
    def clear_processed(self):
        """Clear all processed messages (use with caution)."""
        with self._file_lock:
            self._processed_messages.clear()
            self._save_processed_messages()
            logger.warning("Cleared all processed messages")

# Global processed message log instance
processed_log = ProcessedMessageLog() 