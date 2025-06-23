import json
import re
import random
from typing import Optional, Dict, Any
from pathlib import Path
from .logger import get_logger

logger = get_logger(__name__)

class QueryRouter:
    """Routes queries to appropriate handlers - FAQ cache or full AI pipeline."""
    
    def __init__(self, faq_cache_path: str = "src/backend/faq_cache.json"):
        self.faq_cache_path = Path(faq_cache_path)
        self.faq_data = self._load_faq_cache()
        
    def _load_faq_cache(self) -> Dict[str, Any]:
        """Load FAQ cache from JSON file."""
        try:
            if self.faq_cache_path.exists():
                with open(self.faq_cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"Loaded FAQ cache with {len(data.get('exact_matches', {}))} exact matches")
                    return data
            else:
                logger.warning(f"FAQ cache file not found at {self.faq_cache_path}")
                return {"exact_matches": {}, "keyword_patterns": {}, "greeting_responses": []}
        except Exception as e:
            logger.error(f"Failed to load FAQ cache: {e}")
            return {"exact_matches": {}, "keyword_patterns": {}, "greeting_responses": []}
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query text for matching."""
        # Convert to lowercase and strip whitespace
        normalized = query.lower().strip()
        
        # Remove common question words at the start
        normalized = re.sub(r'^(what is|what are|who is|who are|how do|how does|can you|could you|please)\s+', '', normalized)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove trailing punctuation
        normalized = re.sub(r'[.!?]+$', '', normalized)
        
        return normalized
    
    def _check_exact_match(self, query: str) -> Optional[str]:
        """Check for exact matches in FAQ cache."""
        normalized_query = self._normalize_query(query)
        exact_matches = self.faq_data.get("exact_matches", {})
        
        # Check direct match
        if normalized_query in exact_matches:
            return exact_matches[normalized_query]
        
        # Check original query as well
        if query.lower().strip() in exact_matches:
            return exact_matches[query.lower().strip()]
        
        return None
    
    def _check_keyword_patterns(self, query: str) -> Optional[str]:
        """Check for keyword pattern matches."""
        normalized_query = self._normalize_query(query)
        keyword_patterns = self.faq_data.get("keyword_patterns", {})
        
        for keyword, response in keyword_patterns.items():
            if keyword.lower() in normalized_query:
                return response
        
        # Check for greeting patterns - but only if the query is primarily a greeting
        greeting_words = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        
        # Use original query (not normalized) for greeting detection
        original_lower = query.lower().strip()
        original_words = original_lower.split()
        
        # Question indicators that suggest this is a real question, not just a greeting
        question_indicators = ['what', 'how', 'when', 'where', 'why', 'who', 'which', 'can', 'could', 'should', 'would', 'is', 'are', 'do', 'does', 'did', 'will', 'would', 'tell', 'explain', 'show', 'find', 'help', 'get', 'give', 'status', 'update', 'latest', '?']
        
        # Only treat as greeting if:
        # 1. Query is very short (1-2 words) AND contains only greeting words, OR  
        # 2. Query contains greeting words but NO question indicators
        is_greeting = False
        
        if len(original_words) <= 2:
            # Very short query - check if it's only greetings and pleasantries
            non_greeting_words = [word for word in original_words if word not in greeting_words and word not in ['there', 'everyone', 'all']]
            is_greeting = len(non_greeting_words) == 0
        else:
            # Longer query - only treat as greeting if it has NO question indicators
            has_greeting = any(greeting in original_lower for greeting in greeting_words)
            has_question_content = any(indicator in original_lower for indicator in question_indicators)
            
            # Must have greeting AND no question content to be treated as greeting
            is_greeting = has_greeting and not has_question_content
        
        if is_greeting:
            greeting_responses = self.faq_data.get("greeting_responses", [])
            if greeting_responses:
                return random.choice(greeting_responses)
        
        return None
    
    def _check_simple_question_patterns(self, query: str) -> Optional[str]:
        """Check for simple question patterns that can be answered without AI."""
        normalized_query = self._normalize_query(query)
        
        # Define pattern
        if re.match(r'^(define|what does .* mean|meaning of)', normalized_query):
            return "I'd be happy to help you find definitions! However, I need to search through your server's content to provide accurate definitions based on your context. Let me search for you..."
        
        # Version or status questions
        if re.match(r'^(version|status|health)', normalized_query):
            return "I'm running smoothly and ready to help! Use `/ask` followed by your question to search through your server's knowledge base."
        
        return None
    
    def route_query(self, query: str, user_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Route a query to the appropriate handler.
        
        Args:
            query: User's question
            user_id: Optional user ID for logging
            
        Returns:
            Dictionary with response if handled by router, None if should use AI pipeline
        """
        if not query or not query.strip():
            return {
                "answer": "I'd be happy to help! Please ask me a question about your server's content.",
                "source": "router",
                "confidence": 1.0,
                "citations": []
            }
        
        # Log the routing attempt
        logger.debug(f"Routing query from user {user_id}: {query[:100]}...")
        
        # Check exact matches first
        exact_response = self._check_exact_match(query)
        if exact_response:
            logger.info(f"Query routed to FAQ exact match for user {user_id}")
            return {
                "answer": exact_response,
                "source": "faq_exact",
                "confidence": 1.0,
                "citations": []
            }
        
        # Check keyword patterns
        keyword_response = self._check_keyword_patterns(query)
        if keyword_response:
            logger.info(f"Query routed to FAQ keyword match for user {user_id}")
            return {
                "answer": keyword_response,
                "source": "faq_keyword",
                "confidence": 0.9,
                "citations": []
            }
        
        # Check simple patterns
        simple_response = self._check_simple_question_patterns(query)
        if simple_response:
            logger.info(f"Query routed to simple pattern match for user {user_id}")
            return {
                "answer": simple_response,
                "source": "simple_pattern",
                "confidence": 0.8,
                "citations": []
            }
        
        # No match found - route to AI pipeline
        logger.debug(f"Query will be routed to AI pipeline for user {user_id}")
        return None
    
    def add_faq_entry(self, question: str, answer: str, entry_type: str = "exact") -> bool:
        """
        Add a new FAQ entry (for admin use).
        
        Args:
            question: Question text
            answer: Answer text
            entry_type: Type of entry ("exact" or "keyword")
            
        Returns:
            True if successfully added
        """
        try:
            normalized_question = self._normalize_query(question)
            
            if entry_type == "exact":
                self.faq_data["exact_matches"][normalized_question] = answer
            elif entry_type == "keyword":
                self.faq_data["keyword_patterns"][normalized_question] = answer
            else:
                logger.error(f"Invalid entry type: {entry_type}")
                return False
            
            # Save back to file
            with open(self.faq_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.faq_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Added FAQ entry: {question} -> {answer[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add FAQ entry: {e}")
            return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get FAQ cache statistics."""
        return {
            "exact_matches": len(self.faq_data.get("exact_matches", {})),
            "keyword_patterns": len(self.faq_data.get("keyword_patterns", {})),
            "greeting_responses": len(self.faq_data.get("greeting_responses", []))
        }

# Global query router instance
query_router = QueryRouter() 