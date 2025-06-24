"""
VITA v7.1 Context Budget Allocator

This module provides intelligent context window management:
- Multi-factor chunk scoring and ranking
- Role-aware content prioritization
- Token budget optimization
- Query intent detection
"""

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math

from .logger import get_logger

logger = get_logger(__name__)

class ContextBudgetAllocator:
    """
    Intelligent context window manager that optimizes the selection
    of content chunks based on relevance, freshness, and user context.
    """
    
    def __init__(self, max_tokens: int = 80000):
        self.max_tokens = max_tokens
        self.token_safety_margin = 0.85  # Use 85% of available tokens
        
        # Query intent patterns for depth control
        self.deep_dive_patterns = [
            r'\b(deep dive|in detail|technical specs|comprehensive|thorough)\b',
            r'\b(explain how|walk me through|step by step)\b',
            r'\b(architecture|implementation|under the hood)\b'
        ]
        
        self.summary_patterns = [
            r'\b(summary|overview|brief|quick|tldr)\b',
            r'\b(what is|what are|tell me about)\b',
            r'\b(high level|executive summary)\b'
        ]
        
        # Role-based viewpoint preferences
        self.role_viewpoint_mapping = {
            'executive': ['executive', 'actionable', 'risk_legal'],
            'ceo': ['executive', 'actionable', 'risk_legal'],
            'founder': ['executive', 'actionable', 'technical'],
            'director': ['executive', 'actionable'],
            'vp': ['executive', 'actionable'],
            'manager': ['actionable', 'executive', 'technical'],
            'developer': ['technical', 'qa_synthetic', 'actionable'],
            'engineer': ['technical', 'qa_synthetic', 'actionable'],
            'architect': ['technical', 'executive', 'risk_legal'],
            'devops': ['technical', 'actionable'],
            'legal': ['risk_legal', 'executive', 'actionable'],
            'compliance': ['risk_legal', 'executive'],
            'security': ['risk_legal', 'technical'],
            'project manager': ['actionable', 'executive', 'technical'],
            'scrum master': ['actionable', 'technical']
        }
    
    async def allocate_context(self, candidates: List[Dict], user_context: Dict) -> List[Dict]:
        """
        Intelligently allocate context budget across candidate chunks.
        
        Args:
            candidates: List of candidate content chunks with metadata
            user_context: User context including roles, query intent, channel
            
        Returns:
            Optimally selected and ordered chunks within token budget
        """
        try:
            if not candidates:
                return []
            
            logger.debug(f"Allocating context budget for {len(candidates)} candidates")
            
            # Detect query intent and adjust budget
            query_intent = self._detect_query_intent(user_context.get('question', ''))
            effective_budget = self._calculate_effective_budget(query_intent)
            
            # Score all candidate chunks
            scored_chunks = []
            for chunk in candidates:
                score = self._calculate_value_score(chunk, user_context, query_intent)
                scored_chunks.append((score, chunk))
            
            # Sort by value score (highest first)
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            
            # Pack context within budget using smart selection
            selected_chunks = self._pack_context_budget(scored_chunks, effective_budget, user_context)
            
            # Apply final optimizations
            optimized_chunks = self._optimize_chunk_order(selected_chunks, user_context)
            
            logger.info(f"Selected {len(optimized_chunks)} chunks from {len(candidates)} candidates")
            return optimized_chunks
            
        except Exception as e:
            logger.error(f"Failed to allocate context budget: {e}")
            # Fallback: return top chunks by basic relevance
            return candidates[:10]
    
    def _detect_query_intent(self, question: str) -> Dict[str, any]:
        """Detect user query intent to adjust context allocation."""
        try:
            question_lower = question.lower()
            
            intent = {
                'depth': 'standard',
                'focus': 'general',
                'urgency': 'normal'
            }
            
            # Detect depth preference
            if any(re.search(pattern, question_lower) for pattern in self.deep_dive_patterns):
                intent['depth'] = 'deep'
            elif any(re.search(pattern, question_lower) for pattern in self.summary_patterns):
                intent['depth'] = 'summary'
            
            # Detect focus area
            if any(word in question_lower for word in ['technical', 'code', 'api', 'architecture']):
                intent['focus'] = 'technical'
            elif any(word in question_lower for word in ['business', 'strategy', 'revenue', 'decision']):
                intent['focus'] = 'business'
            elif any(word in question_lower for word in ['action', 'todo', 'next steps', 'implement']):
                intent['focus'] = 'actionable'
            elif any(word in question_lower for word in ['risk', 'compliance', 'legal', 'security']):
                intent['focus'] = 'risk'
            
            # Detect urgency
            if any(word in question_lower for word in ['urgent', 'asap', 'immediately', 'critical']):
                intent['urgency'] = 'high'
            
            return intent
            
        except Exception as e:
            logger.error(f"Failed to detect query intent: {e}")
            return {'depth': 'standard', 'focus': 'general', 'urgency': 'normal'}
    
    def _calculate_effective_budget(self, query_intent: Dict) -> int:
        """Calculate effective token budget based on query intent."""
        base_budget = int(self.max_tokens * self.token_safety_margin)
        
        # Adjust based on depth preference
        if query_intent['depth'] == 'deep':
            return int(base_budget * 1.2)  # 20% more tokens for deep dives
        elif query_intent['depth'] == 'summary':
            return int(base_budget * 0.7)  # 30% fewer tokens for summaries
        
        return base_budget
    
    def _calculate_value_score(self, chunk: Dict, user_context: Dict, query_intent: Dict) -> float:
        """Calculate comprehensive value score for a chunk."""
        try:
            metadata = chunk.get('metadata', {})
            
            # Base relevance from vector search
            relevance_score = chunk.get('score', 0.5)
            
            # Freshness score (newer = better, with decay)
            freshness_score = self._calculate_freshness_score(metadata.get('timestamp'))
            
            # Category weight (CRITICAL > HIGH > MEDIUM > LOW)
            category_weight = self._get_category_weight(metadata.get('relevance_category', 'MEDIUM'))
            
            # Viewpoint matching bonus
            viewpoint_bonus = self._calculate_viewpoint_bonus(metadata, user_context)
            
            # Channel relevance bonus
            channel_bonus = self._calculate_channel_bonus(metadata, user_context)
            
            # Query intent alignment
            intent_bonus = self._calculate_intent_bonus(metadata, query_intent)
            
            # Content type bonus
            content_type_bonus = self._calculate_content_type_bonus(metadata)
            
            # Role-specific retrieval weight
            retrieval_weight = metadata.get('retrieval_weight', 1.0)
            
            # Combine all factors with weights
            final_score = (
                relevance_score * 0.35 +
                freshness_score * 0.15 +
                category_weight * 0.20 +
                viewpoint_bonus * 0.10 +
                channel_bonus * 0.05 +
                intent_bonus * 0.10 +
                content_type_bonus * 0.05
            ) * retrieval_weight
            
            return min(2.0, max(0.0, final_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate value score: {e}")
            return 0.5
    
    def _calculate_freshness_score(self, timestamp_str: Optional[str]) -> float:
        """Calculate freshness score with exponential decay."""
        try:
            if not timestamp_str:
                return 0.5  # Neutral score for unknown timestamps
            
            # Parse timestamp
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                return 0.5
            
            # Calculate age in days
            age_days = (datetime.utcnow() - timestamp.replace(tzinfo=None)).total_seconds() / 86400
            
            # Exponential decay: score = e^(-age/30) 
            # Content is "fresh" for about 30 days
            freshness_score = math.exp(-age_days / 30)
            
            return min(1.0, max(0.1, freshness_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate freshness score: {e}")
            return 0.5
    
    def _get_category_weight(self, category: str) -> float:
        """Get weight multiplier based on content category."""
        weights = {
            'CRITICAL': 2.0,
            'HIGH': 1.5,
            'MEDIUM': 1.0,
            'LOW': 0.5,
            'IRRELEVANT': 0.1
        }
        return weights.get(category, 1.0)
    
    def _calculate_viewpoint_bonus(self, metadata: Dict, user_context: Dict) -> float:
        """Calculate bonus for viewpoint matching user roles."""
        try:
            user_roles = user_context.get('roles', [])
            viewpoint = metadata.get('viewpoint')
            
            if not viewpoint or not user_roles:
                return 0.5
            
            # Check if viewpoint matches user role preferences
            role_bonus = 0.0
            for role in user_roles:
                role_lower = role.lower()
                if role_lower in self.role_viewpoint_mapping:
                    preferred_viewpoints = self.role_viewpoint_mapping[role_lower]
                    if viewpoint in preferred_viewpoints:
                        # Higher bonus for earlier positions in preference list
                        position_bonus = (len(preferred_viewpoints) - preferred_viewpoints.index(viewpoint)) / len(preferred_viewpoints)
                        role_bonus = max(role_bonus, position_bonus)
            
            return min(1.0, max(0.0, role_bonus))
            
        except Exception as e:
            logger.error(f"Failed to calculate viewpoint bonus: {e}")
            return 0.5
    
    def _calculate_channel_bonus(self, metadata: Dict, user_context: Dict) -> float:
        """Calculate bonus for content from relevant channels."""
        try:
            content_channel = metadata.get('channel_id', '')
            query_channel = user_context.get('channel_id', '')
            
            if not content_channel or not query_channel:
                return 0.5
            
            # Exact channel match
            if content_channel == query_channel:
                return 1.0
            
            # Related channel bonus (same category)
            content_channel_lower = content_channel.lower()
            query_channel_lower = query_channel.lower()
            
            # Check for similar channel types
            if any(keyword in content_channel_lower and keyword in query_channel_lower 
                   for keyword in ['dev', 'prod', 'tech', 'business', 'general']):
                return 0.7
            
            return 0.3  # Different channel penalty
            
        except Exception as e:
            logger.error(f"Failed to calculate channel bonus: {e}")
            return 0.5
    
    def _calculate_intent_bonus(self, metadata: Dict, query_intent: Dict) -> float:
        """Calculate bonus based on query intent alignment."""
        try:
            viewpoint = metadata.get('viewpoint', '')
            content_type = metadata.get('content_type', '')
            
            focus = query_intent.get('focus', 'general')
            depth = query_intent.get('depth', 'standard')
            
            bonus = 0.5  # Base bonus
            
            # Focus alignment
            if focus == 'technical' and viewpoint == 'technical':
                bonus += 0.4
            elif focus == 'business' and viewpoint == 'executive':
                bonus += 0.4
            elif focus == 'actionable' and viewpoint == 'actionable':
                bonus += 0.4
            elif focus == 'risk' and viewpoint == 'risk_legal':
                bonus += 0.4
            
            # Depth alignment
            if depth == 'deep' and viewpoint == 'technical':
                bonus += 0.2
            elif depth == 'summary' and viewpoint == 'executive':
                bonus += 0.2
            elif depth == 'standard' and viewpoint in ['qa_synthetic', 'actionable']:
                bonus += 0.1
            
            # Content type bonuses
            if content_type == 'synthetic_qa' and depth == 'standard':
                bonus += 0.2
            elif content_type == 'document_synthesis' and depth == 'deep':
                bonus += 0.3
            
            return min(1.0, max(0.0, bonus))
            
        except Exception as e:
            logger.error(f"Failed to calculate intent bonus: {e}")
            return 0.5
    
    def _calculate_content_type_bonus(self, metadata: Dict) -> float:
        """Calculate bonus based on content type value."""
        content_type = metadata.get('content_type', '')
        
        type_bonuses = {
            'document_synthesis': 0.8,  # Cross-document insights are valuable
            'viewpoint_analysis': 0.7,  # Specialized analysis
            'synthetic_qa': 0.6,        # Ready-to-use Q&A
            'original_content': 0.5     # Base content
        }
        
        return type_bonuses.get(content_type, 0.5)
    
    def _pack_context_budget(self, scored_chunks: List[Tuple[float, Dict]], 
                           budget: int, user_context: Dict) -> List[Dict]:
        """Pack chunks into context budget using smart selection."""
        try:
            selected_chunks = []
            current_tokens = 0
            
            # Reserve tokens for system prompt and response
            system_overhead = 2000
            available_budget = budget - system_overhead
            
            # First pass: select high-value chunks
            for score, chunk in scored_chunks:
                chunk_tokens = self._estimate_tokens(chunk['content'])
                
                if current_tokens + chunk_tokens <= available_budget:
                    selected_chunks.append(chunk)
                    current_tokens += chunk_tokens
                else:
                    # Try to fit smaller chunks in remaining space
                    remaining_budget = available_budget - current_tokens
                    if chunk_tokens <= remaining_budget and remaining_budget > 100:
                        # Truncate chunk if it's close to fitting
                        truncated_content = self._truncate_content(chunk['content'], remaining_budget)
                        if truncated_content:
                            truncated_chunk = chunk.copy()
                            truncated_chunk['content'] = truncated_content
                            truncated_chunk['metadata'] = chunk['metadata'].copy()
                            truncated_chunk['metadata']['truncated'] = True
                            selected_chunks.append(truncated_chunk)
                            break
            
            logger.debug(f"Packed {len(selected_chunks)} chunks using {current_tokens}/{available_budget} tokens")
            return selected_chunks
            
        except Exception as e:
            logger.error(f"Failed to pack context budget: {e}")
            return [chunk for _, chunk in scored_chunks[:10]]  # Fallback
    
    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for content."""
        # Rough estimation: ~4 characters per token for English text
        return len(content) // 4 + 50  # Add small buffer
    
    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Intelligently truncate content to fit token budget."""
        try:
            max_chars = max_tokens * 4  # Convert tokens to characters
            
            if len(content) <= max_chars:
                return content
            
            # Try to truncate at sentence boundaries
            sentences = content.split('. ')
            truncated = ""
            
            for sentence in sentences:
                if len(truncated + sentence + '. ') <= max_chars - 20:  # Leave margin
                    truncated += sentence + '. '
                else:
                    break
            
            if truncated:
                return truncated.strip() + "..."
            
            # Fallback: character truncation
            return content[:max_chars-3] + "..."
            
        except Exception as e:
            logger.error(f"Failed to truncate content: {e}")
            return content[:max_tokens*3]  # Conservative fallback
    
    def _optimize_chunk_order(self, chunks: List[Dict], user_context: Dict) -> List[Dict]:
        """Optimize the order of selected chunks for better context flow."""
        try:
            if len(chunks) <= 1:
                return chunks
            
            # Group chunks by type and viewpoint
            grouped_chunks = self._group_chunks_by_type(chunks)
            
            # Order groups by importance for the query
            ordered_chunks = []
            
            # 1. Executive summaries first (if user has executive role)
            user_roles = [role.lower() for role in user_context.get('roles', [])]
            if any(role in ['executive', 'ceo', 'director', 'vp'] for role in user_roles):
                ordered_chunks.extend(grouped_chunks.get('executive', []))
            
            # 2. Q&A pairs for quick answers
            ordered_chunks.extend(grouped_chunks.get('qa_synthetic', [])[:3])  # Limit Q&A
            
            # 3. Technical details (if relevant)
            if any(role in ['developer', 'engineer', 'architect'] for role in user_roles):
                ordered_chunks.extend(grouped_chunks.get('technical', []))
            
            # 4. Actionable items
            ordered_chunks.extend(grouped_chunks.get('actionable', []))
            
            # 5. Other viewpoints
            for viewpoint, chunks_list in grouped_chunks.items():
                if viewpoint not in ['executive', 'qa_synthetic', 'technical', 'actionable']:
                    ordered_chunks.extend(chunks_list)
            
            # Remove duplicates while preserving order
            seen_content = set()
            final_chunks = []
            for chunk in ordered_chunks:
                content_hash = hash(chunk['content'][:100])  # Hash first 100 chars
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    final_chunks.append(chunk)
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"Failed to optimize chunk order: {e}")
            return chunks
    
    def _group_chunks_by_type(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """Group chunks by viewpoint type."""
        grouped = {}
        
        for chunk in chunks:
            viewpoint = chunk.get('metadata', {}).get('viewpoint', 'other')
            if viewpoint not in grouped:
                grouped[viewpoint] = []
            grouped[viewpoint].append(chunk)
        
        return grouped

# Global instance
context_allocator = ContextBudgetAllocator()
