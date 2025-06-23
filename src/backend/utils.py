import re
from typing import List, Dict, Any
from .logger import get_logger

logger = get_logger(__name__)

def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    return text

def redact_sensitive_info(text: str) -> str:
    """
    Redact potentially sensitive information from text.
    
    Args:
        text: Text to redact
        
    Returns:
        Redacted text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Redact email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]', text)
    
    # Redact phone numbers (basic patterns)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]', text)
    text = re.sub(r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]', text)
    
    # Redact URLs (but keep domain for context)
    text = re.sub(r'https?://[^\s<>"]+', '[URL_REDACTED]', text)
    
    return text

def split_text_for_embedding(text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into chunks suitable for embedding.
    
    Args:
        text: Text to split
        max_chunk_size: Maximum characters per chunk
        overlap: Characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text) <= max_chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_size
        
        # If we're not at the end, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 200 characters
            search_start = max(start + max_chunk_size - 200, start)
            sentence_end = -1
            
            for i in range(end - 1, search_start - 1, -1):
                if text[i] in '.!?':
                    # Make sure it's not an abbreviation
                    if i < len(text) - 1 and text[i + 1].isspace():
                        sentence_end = i + 1
                        break
            
            if sentence_end > start:
                end = sentence_end
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = max(end - overlap, start + 1)
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks

def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize metadata for Pinecone storage.
    
    Args:
        metadata: Raw metadata dict
        
    Returns:
        Sanitized metadata dict
    """
    sanitized = {}
    
    for key, value in metadata.items():
        if value is None:
            sanitized[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, list):
            # Convert list to string representation
            sanitized[key] = str(value)
        else:
            # Convert other types to string
            sanitized[key] = str(value)
    
    return sanitized

def extract_message_url(guild_id: str, channel_id: str, message_id: str) -> str:
    """
    Generate Discord message URL.
    
    Args:
        guild_id: Discord guild ID
        channel_id: Discord channel ID
        message_id: Discord message ID
        
    Returns:
        Discord message URL
    """
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}" 