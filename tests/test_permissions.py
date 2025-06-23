"""
Unit tests for permissions module.
"""
import pytest
from src.backend.permissions import has_permission, ROLE_PERMISSIONS


class TestHasPermission:
    """Test the has_permission function."""
    
    def test_admin_has_all_permissions(self):
        """Test that admin role has all permissions."""
        admin_roles = ["Admin", "@Admin"]
        
        for role in admin_roles:
            assert has_permission([role], "query")
            assert has_permission([role], "ingest")
            assert has_permission([role], "view_logs")
    
    def test_moderator_permissions(self):
        """Test moderator role permissions."""
        moderator_roles = ["Moderator", "@Moderator"]
        
        for role in moderator_roles:
            assert has_permission([role], "query")
            assert has_permission([role], "ingest")
            # Moderators should not have admin-level permissions by default
    
    def test_everyone_query_permission(self):
        """Test that everyone can query."""
        assert has_permission(["@everyone"], "query")
        assert has_permission(["Member"], "query")
        assert has_permission(["User"], "query")
    
    def test_no_roles_default_permission(self):
        """Test behavior with no roles."""
        # Should default to allowing basic query
        assert has_permission([], "query")
    
    def test_case_insensitive_roles(self):
        """Test that role checking is case insensitive."""
        assert has_permission(["admin"], "query")
        assert has_permission(["ADMIN"], "query") 
        assert has_permission(["Admin"], "query")
        assert has_permission(["moderator"], "ingest")
        assert has_permission(["MODERATOR"], "ingest")
    
    def test_role_with_at_symbol(self):
        """Test roles with @ symbol are handled correctly."""
        assert has_permission(["@Admin"], "query")
        assert has_permission(["@Moderator"], "ingest")
        assert has_permission(["@everyone"], "query")
    
    def test_multiple_roles(self):
        """Test permission checking with multiple roles."""
        # User with multiple roles should get highest permission
        mixed_roles = ["Member", "Moderator", "SomeOtherRole"]
        assert has_permission(mixed_roles, "query")
        assert has_permission(mixed_roles, "ingest")
        
        # Admin among other roles
        admin_mixed = ["Member", "Admin", "User"]
        assert has_permission(admin_mixed, "query")
        assert has_permission(admin_mixed, "ingest")
        assert has_permission(admin_mixed, "view_logs")
    
    def test_invalid_permission(self):
        """Test behavior with invalid permission names."""
        # Should handle gracefully and return False for unknown permissions
        assert not has_permission(["Admin"], "invalid_permission")
        assert not has_permission(["Admin"], "")
        assert not has_permission(["Admin"], None)
    
    def test_role_permissions_structure(self):
        """Test that ROLE_PERMISSIONS dictionary is properly structured."""
        # Verify the structure exists and has expected keys
        assert isinstance(ROLE_PERMISSIONS, dict)
        
        # Common roles should exist
        expected_roles = ["admin", "moderator", "everyone"]
        for role in expected_roles:
            assert any(role in key.lower() for key in ROLE_PERMISSIONS.keys())
        
        # Each role should have a list of permissions
        for role, permissions in ROLE_PERMISSIONS.items():
            assert isinstance(permissions, list)
            # Each permission should be a string
            for permission in permissions:
                assert isinstance(permission, str)


class TestPermissionEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_none_roles(self):
        """Test behavior when roles is None."""
        # Should handle gracefully
        result = has_permission(None, "query")
        # Should either return False or handle gracefully
        assert isinstance(result, bool)
    
    def test_empty_string_roles(self):
        """Test behavior with empty string roles."""
        assert has_permission([""], "query") == has_permission([], "query")
    
    def test_whitespace_only_roles(self):
        """Test behavior with whitespace-only roles."""
        result = has_permission(["   ", "\t", "\n"], "query")
        assert isinstance(result, bool)
    
    def test_special_characters_in_roles(self):
        """Test roles with special characters."""
        special_roles = ["Role#1", "Role@123", "Role$pecial", "Role%test"]
        for role in special_roles:
            result = has_permission([role], "query")
            assert isinstance(result, bool)
    
    def test_very_long_role_names(self):
        """Test with very long role names."""
        long_role = "A" * 1000  # Very long role name
        result = has_permission([long_role], "query")
        assert isinstance(result, bool)
    
    def test_unicode_in_roles(self):
        """Test roles with Unicode characters."""
        unicode_roles = ["Röle", "角色", "роль", "🎭Admin"]
        for role in unicode_roles:
            result = has_permission([role], "query")
            assert isinstance(result, bool) 