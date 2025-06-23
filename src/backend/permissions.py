from typing import List, Dict, Any
from .logger import get_logger

logger = get_logger(__name__)

class PermissionManager:
    """Manages role-based permissions for knowledge access."""
    
    def __init__(self):
        # Default permission levels (can be extended)
        self.permission_hierarchy = {
            "admin": 100,
            "moderator": 75,
            "member": 50,
            "guest": 25,
            "@everyone": 10
        }
    
    def create_permission_filter(self, user_roles: List[str], channel_id: str) -> Dict[str, Any]:
        """
        Create a Pinecone filter for role-based access control.
        
        Args:
            user_roles: List of user's roles
            channel_id: Current channel ID
            
        Returns:
            Pinecone filter dictionary
        """
        if not user_roles:
            user_roles = ["@everyone"]
        
        # Get highest permission level
        max_permission = self._get_max_permission_level(user_roles)
        
        # Create filter conditions
        filter_conditions = {
            "$or": [
                # Allow access to messages from the same channel
                {"channel_id": channel_id},
                # Allow access to public content (no specific roles required)
                {"required_roles": {"$exists": False}},
                # Allow access based on role permissions
                {"required_permission_level": {"$lte": max_permission}}
            ]
        }
        
        logger.debug(f"Created permission filter for roles {user_roles}: {filter_conditions}")
        return filter_conditions
    
    def _get_max_permission_level(self, user_roles: List[str]) -> int:
        """
        Get the maximum permission level for a user's roles.
        
        Args:
            user_roles: List of user's roles
            
        Returns:
            Maximum permission level
        """
        max_level = 0
        
        for role in user_roles:
            role_level = self.permission_hierarchy.get(role.lower(), 0)
            max_level = max(max_level, role_level)
        
        # Ensure minimum permission level
        if max_level == 0:
            max_level = self.permission_hierarchy.get("@everyone", 10)
        
        return max_level
    
    def can_access_content(self, user_roles: List[str], content_roles: List[str] = None) -> bool:
        """
        Check if user can access content based on roles.
        
        Args:
            user_roles: User's roles
            content_roles: Roles required to access content
            
        Returns:
            True if user can access content
        """
        if not content_roles:
            return True  # Public content
        
        if not user_roles:
            user_roles = ["@everyone"]
        
        user_permission = self._get_max_permission_level(user_roles)
        required_permission = self._get_max_permission_level(content_roles)
        
        return user_permission >= required_permission
    
    def add_permission_metadata(self, metadata: Dict[str, Any], user_roles: List[str] = None) -> Dict[str, Any]:
        """
        Add permission-related metadata to content.
        
        Args:
            metadata: Existing metadata
            user_roles: Roles associated with the content creator
            
        Returns:
            Metadata with permission information
        """
        if user_roles:
            metadata["required_roles"] = user_roles
            metadata["required_permission_level"] = self._get_max_permission_level(user_roles)
        
        return metadata

# Global permission manager instance
permission_manager = PermissionManager() 