"""
Unit tests for backend utility functions.
"""
import pytest
from src.backend.utils import clean_text, redact_sensitive_info, split_text_for_embedding


class TestCleanText:
    """Test the clean_text function."""
    
    def test_remove_extra_whitespace(self):
        """Test removal of extra whitespace."""
        assert clean_text("  hello   world  ") == "hello world"
        assert clean_text("hello\n\n\nworld") == "hello world"
        assert clean_text("hello\t\tworld") == "hello world"
    
    def test_remove_control_characters(self):
        """Test removal of control characters."""
        assert clean_text("hello\x07world") == "hello world"
        assert clean_text("text\x00with\x01control") == "text with control"
    
    def test_normalize_unicode(self):
        """Test Unicode normalization."""
        assert clean_text("café") == "café"  # Should normalize Unicode
    
    def test_empty_and_whitespace_only(self):
        """Test edge cases with empty or whitespace-only strings."""
        assert clean_text("") == ""
        assert clean_text("   ") == ""
        assert clean_text("\n\t\r") == ""
    
    def test_preserve_normal_text(self):
        """Test that normal text is preserved."""
        normal_text = "This is normal text with punctuation!"
        assert clean_text(normal_text) == normal_text


class TestRedactSensitiveInfo:
    """Test the redact_sensitive_info function."""
    
    def test_redact_emails(self):
        """Test email redaction."""
        text = "Contact me at john.doe@example.com for more info"
        result = redact_sensitive_info(text)
        assert "john.doe@example.com" not in result
        assert "[EMAIL]" in result
    
    def test_redact_phone_numbers(self):
        """Test phone number redaction."""
        # Test various phone number formats
        test_cases = [
            "Call me at (555) 123-4567",
            "My number is 555-123-4567",
            "Phone: +1-555-123-4567",
            "Text 5551234567"
        ]
        
        for text in test_cases:
            result = redact_sensitive_info(text)
            assert "[PHONE]" in result
    
    def test_redact_urls(self):
        """Test URL redaction."""
        text = "Visit https://example.com or http://test.org"
        result = redact_sensitive_info(text)
        assert "https://example.com" not in result
        assert "http://test.org" not in result
        assert "[URL]" in result
    
    def test_preserve_normal_text(self):
        """Test that normal text without sensitive info is preserved."""
        normal_text = "This is just normal text without any sensitive information."
        result = redact_sensitive_info(normal_text)
        assert result == normal_text
    
    def test_multiple_redactions(self):
        """Test text with multiple types of sensitive information."""
        text = "Email me at test@example.com or call (555) 123-4567. Visit https://mysite.com"
        result = redact_sensitive_info(text)
        assert "[EMAIL]" in result
        assert "[PHONE]" in result
        assert "[URL]" in result
        assert "test@example.com" not in result
        assert "(555) 123-4567" not in result
        assert "https://mysite.com" not in result


class TestSplitTextForEmbedding:
    """Test the split_text_for_embedding function."""
    
    def test_short_text_no_split(self):
        """Test that short text is not split."""
        short_text = "This is a short text."
        chunks = split_text_for_embedding(short_text, max_chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0] == short_text
    
    def test_long_text_split(self):
        """Test that long text is properly split."""
        # Create a long text that exceeds chunk size
        long_text = "This is a sentence. " * 100  # 2000+ characters
        chunks = split_text_for_embedding(long_text, max_chunk_size=500)
        
        assert len(chunks) > 1
        # Each chunk should be <= max_chunk_size
        for chunk in chunks:
            assert len(chunk) <= 500
    
    def test_split_on_sentences(self):
        """Test that text is split on sentence boundaries when possible."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = split_text_for_embedding(text, max_chunk_size=30)
        
        # Should split on sentence boundaries
        assert all(chunk.strip().endswith('.') for chunk in chunks if chunk.strip())
    
    def test_overlap_functionality(self):
        """Test that overlapping chunks work correctly."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks = split_text_for_embedding(text, max_chunk_size=30, overlap=10)
        
        assert len(chunks) >= 2
        # There should be some overlap between consecutive chunks
        if len(chunks) > 1:
            # Check that there's some shared content (basic overlap test)
            assert len(chunks[0]) > 0 and len(chunks[1]) > 0
    
    def test_empty_text(self):
        """Test edge case with empty text."""
        chunks = split_text_for_embedding("", max_chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0] == ""
    
    def test_whitespace_only_text(self):
        """Test edge case with whitespace-only text."""
        chunks = split_text_for_embedding("   \n\t  ", max_chunk_size=1000)
        assert len(chunks) == 1


# Integration test for the full pipeline
class TestUtilsPipeline:
    """Test the combination of utility functions."""
    
    def test_full_processing_pipeline(self):
        """Test the complete text processing pipeline."""
        # Raw text with various issues
        raw_text = """
        Hello,   this is a test message.
        
        Please contact me at john@example.com or call (555) 123-4567.
        Visit our website at https://example.com for more info.
        
        This text has extra   spaces and    control characters.\x07
        """
        
        # Step 1: Clean the text
        cleaned = clean_text(raw_text)
        assert "\x07" not in cleaned
        assert "   " not in cleaned  # Extra spaces removed
        
        # Step 2: Redact sensitive information
        redacted = redact_sensitive_info(cleaned)
        assert "john@example.com" not in redacted
        assert "(555) 123-4567" not in redacted
        assert "https://example.com" not in redacted
        assert "[EMAIL]" in redacted
        assert "[PHONE]" in redacted
        assert "[URL]" in redacted
        
        # Step 3: Split for embedding
        chunks = split_text_for_embedding(redacted, max_chunk_size=100)
        assert len(chunks) >= 1
        assert all(len(chunk) <= 100 for chunk in chunks) 