"""
Integration tests for the FastAPI backend API.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

# We'll need to import the FastAPI app
from src.backend.api import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Provide authentication headers for protected endpoints."""
    return {"X-API-Key": "vita-secure-backend-key-1719151463"}


class TestHealthEndpoint:
    """Test the health check endpoint."""
    
    def test_health_check_success(self, client):
        """Test successful health check."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert data["status"] == "healthy"
    
    def test_health_check_structure(self, client):
        """Test that health check returns expected structure."""
        response = client.get("/health")
        data = response.json()
        
        # Verify the expected structure
        assert isinstance(data["services"], dict)
        expected_services = ["api", "openai", "pinecone"]
        for service in expected_services:
            assert service in data["services"]


class TestQueryEndpoint:
    """Test the query endpoint."""
    
    def test_query_without_auth(self, client):
        """Test query endpoint without authentication."""
        response = client.post("/query", 
                             json={"query": "test query", "user_roles": ["@everyone"]})
        assert response.status_code == 401  # Should require authentication
    
    @patch('src.backend.api.llm_client')
    def test_query_with_auth(self, mock_llm_client, client, auth_headers):
        """Test query endpoint with proper authentication."""
        # Mock the LLM client response
        mock_llm_client.generate_response.return_value = "Test response"
        
        response = client.post("/query",
                             headers=auth_headers,
                             json={"query": "test query", "user_roles": ["@everyone"]})
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
    
    def test_query_invalid_json(self, client, auth_headers):
        """Test query endpoint with invalid JSON."""
        response = client.post("/query",
                             headers=auth_headers,
                             json={})  # Missing required fields
        
        assert response.status_code == 422  # Validation error
    
    def test_query_permission_denied(self, client, auth_headers):
        """Test query with insufficient permissions."""
        # This test depends on the actual permission implementation
        response = client.post("/query",
                             headers=auth_headers,
                             json={"query": "test", "user_roles": []})
        
        # Should either work (if everyone can query) or return 403
        assert response.status_code in [200, 403]


class TestIngestEndpoint:
    """Test the ingestion endpoints."""
    
    def test_ingest_without_auth(self, client):
        """Test ingest endpoint without authentication."""
        message_data = {
            "message_id": "test123",
            "channel_id": "channel123",
            "user_id": "user123",
            "content": "Test message",
            "timestamp": "2024-01-01T12:00:00Z",
            "attachments": [],
            "thread_id": None,
            "roles": ["@everyone"]
        }
        
        response = client.post("/ingest", json=message_data)
        assert response.status_code == 401
    
    @patch('src.backend.api.ingest_message')
    def test_ingest_with_auth(self, mock_ingest, client, auth_headers):
        """Test ingest endpoint with proper authentication."""
        # Mock the ingestion function
        mock_ingest.return_value = {"status": "success"}
        
        message_data = {
            "message_id": "test123",
            "channel_id": "channel123", 
            "user_id": "user123",
            "content": "Test message",
            "timestamp": "2024-01-01T12:00:00Z",
            "attachments": [],
            "thread_id": None,
            "roles": ["@everyone"]
        }
        
        response = client.post("/ingest", 
                             headers=auth_headers,
                             json=message_data)
        
        # Should accept the request (might be async so 202 or 200)
        assert response.status_code in [200, 202]
    
    def test_ingest_invalid_data(self, client, auth_headers):
        """Test ingest endpoint with invalid message data."""
        invalid_data = {
            "message_id": "test123",
            # Missing required fields
        }
        
        response = client.post("/ingest",
                             headers=auth_headers, 
                             json=invalid_data)
        
        assert response.status_code == 422  # Validation error


class TestBatchIngestEndpoint:
    """Test the batch ingestion endpoint."""
    
    def test_batch_ingest_without_auth(self, client):
        """Test batch ingest without authentication."""
        batch_data = {
            "messages": [{
                "message_id": "test123",
                "channel_id": "channel123",
                "user_id": "user123", 
                "content": "Test message",
                "timestamp": "2024-01-01T12:00:00Z",
                "attachments": [],
                "thread_id": None,
                "roles": ["@everyone"]
            }]
        }
        
        response = client.post("/batch_ingest", json=batch_data)
        assert response.status_code == 401
    
    @patch('src.backend.api.progress_tracker')
    def test_batch_ingest_with_auth(self, mock_tracker, client, auth_headers):
        """Test batch ingest with proper authentication."""
        # Mock the progress tracker
        mock_tracker.create_batch.return_value = "test-batch-id"
        
        batch_data = {
            "messages": [{
                "message_id": "test123",
                "channel_id": "channel123",
                "user_id": "user123",
                "content": "Test message", 
                "timestamp": "2024-01-01T12:00:00Z",
                "attachments": [],
                "thread_id": None,
                "roles": ["@everyone"]
            }]
        }
        
        response = client.post("/batch_ingest",
                             headers=auth_headers,
                             json=batch_data)
        
        assert response.status_code == 202  # Accepted for processing
        data = response.json()
        assert "batch_id" in data
    
    def test_batch_ingest_empty_messages(self, client, auth_headers):
        """Test batch ingest with empty message list."""
        batch_data = {"messages": []}
        
        response = client.post("/batch_ingest",
                             headers=auth_headers,
                             json=batch_data)
        
        # Should handle empty batch gracefully
        assert response.status_code in [400, 422]


class TestProgressEndpoint:
    """Test the progress tracking endpoint."""
    
    def test_progress_without_auth(self, client):
        """Test progress endpoint without authentication."""
        response = client.get("/progress/test-batch-id")
        assert response.status_code == 401
    
    @patch('src.backend.api.progress_tracker')
    def test_progress_with_auth(self, mock_tracker, client, auth_headers):
        """Test progress endpoint with authentication."""
        # Mock progress data
        mock_progress = {
            "batch_id": "test-batch-id",
            "total_messages": 10,
            "processed_count": 5,
            "status": "IN_PROGRESS"
        }
        mock_tracker.get_progress.return_value = mock_progress
        
        response = client.get("/progress/test-batch-id",
                            headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "progress" in data
    
    def test_progress_nonexistent_batch(self, client, auth_headers):
        """Test progress endpoint with non-existent batch ID."""
        response = client.get("/progress/nonexistent-batch-id",
                            headers=auth_headers)
        
        # Should return 404 or handle gracefully
        assert response.status_code in [404, 200]


class TestAuthenticationSecurity:
    """Test authentication and security measures."""
    
    def test_invalid_api_key(self, client):
        """Test with invalid API key."""
        invalid_headers = {"X-API-Key": "invalid-key"}
        
        response = client.post("/query",
                             headers=invalid_headers,
                             json={"query": "test", "user_roles": ["@everyone"]})
        
        assert response.status_code == 401
    
    def test_missing_api_key_header(self, client):
        """Test with missing API key header."""
        response = client.post("/query",
                             json={"query": "test", "user_roles": ["@everyone"]})
        
        assert response.status_code == 401
    
    def test_empty_api_key(self, client):
        """Test with empty API key."""
        empty_headers = {"X-API-Key": ""}
        
        response = client.post("/query",
                             headers=empty_headers, 
                             json={"query": "test", "user_roles": ["@everyone"]})
        
        assert response.status_code == 401 