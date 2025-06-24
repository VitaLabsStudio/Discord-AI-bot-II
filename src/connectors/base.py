from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass

@dataclass
class ConnectorItem:
    """
    Standard format for items retrieved from external sources.
    
    This provides a canonical format that all connectors must transform their
    source-specific data into for VITA ingestion.
    """
    # Core identification
    id: str  # Unique identifier within the source system
    source_type: str  # e.g., "notion", "github", "gdrive"
    
    # Content
    title: str
    content: str
    content_type: str  # "text", "document", "code", "wiki"
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    author: Optional[str] = None
    
    # Hierarchy and relationships
    parent_id: Optional[str] = None
    path: Optional[str] = None  # e.g., "workspace/database/page"
    tags: List[str] = None
    
    # Access control
    permissions: Dict[str, Any] = None  # Source-specific permission data
    
    # Source-specific data
    source_metadata: Dict[str, Any] = None

class SourceConnector(ABC):
    """
    Abstract base class for external source connectors.
    
    This defines the standard interface that all VITA source connectors must implement.
    Future connectors for Notion, GitHub, Google Drive, etc. will inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the connector with configuration.
        
        Args:
            config: Configuration dictionary with API keys, endpoints, etc.
        """
        self.config = config
        self.source_type = self.__class__.__name__.lower().replace('connector', '')
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the external source and verify credentials.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection and return status information.
        
        Returns:
            Dictionary with connection status and metadata
        """
        pass
    
    @abstractmethod
    async def fetch_new_items(self, last_sync_timestamp: Optional[datetime] = None) -> AsyncIterator[ConnectorItem]:
        """
        Fetch all new or updated items since the last sync.
        
        Args:
            last_sync_timestamp: Timestamp of last successful sync, None for full sync
            
        Yields:
            ConnectorItem objects in canonical format
        """
        pass
    
    @abstractmethod
    async def fetch_item_by_id(self, item_id: str) -> Optional[ConnectorItem]:
        """
        Fetch a specific item by its ID.
        
        Args:
            item_id: Unique identifier of the item in the source system
            
        Returns:
            ConnectorItem if found, None otherwise
        """
        pass
    
    @abstractmethod
    def to_canonical_format(self, source_item: Any) -> ConnectorItem:
        """
        Convert a source-specific item into VITA's standard ingestion format.
        
        Args:
            source_item: Raw item from the source system
            
        Returns:
            ConnectorItem in canonical format
        """
        pass
    
    async def get_sync_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the sync state and capabilities.
        
        Returns:
            Dictionary with sync metadata
        """
        return {
            "source_type": self.source_type,
            "supports_incremental": True,
            "supports_real_time": False,
            "estimated_items": None,
            "last_sync": None
        }
    
    async def cleanup(self):
        """Clean up any resources used by the connector."""
        pass
    
    def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup() 