"""
VITA v7.1 Adaptive Relevance Engine

This module provides intelligent content relevance classification with:
- Dynamic channel-specific thresholds
- Two-stage LLM cascade (cheap → expensive)
- Learning from user feedback
- Multi-factor relevance scoring
"""

import re
import asyncio
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json

from .database import vita_db
from .llm_client import llm_client
from .logger import get_logger

logger = get_logger(__name__)

class AdaptiveRelevanceEngine:
    """
    Intelligent relevance classification engine that adapts to channel patterns
    and learns from user feedback.
    """
    
    def __init__(self):
        # Business and technical keyword patterns
        self.business_keywords = {
            'strategy', 'revenue', 'budget', 'roi', 'kpi', 'metrics', 'goals',
            'roadmap', 'milestone', 'deadline', 'launch', 'release', 'project',
            'customer', 'client', 'market', 'competition', 'analysis', 'report',
            'meeting', 'decision', 'approval', 'contract', 'agreement', 'legal',
            'compliance', 'risk', 'security', 'privacy', 'gdpr', 'policy'
        }
        
        self.technical_keywords = {
            'api', 'database', 'server', 'deployment', 'production', 'staging',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'cloud', 'infrastructure',
            'code', 'repository', 'git', 'branch', 'merge', 'pull request', 'bug',
            'feature', 'enhancement', 'performance', 'optimization', 'scaling',
            'monitoring', 'logging', 'debugging', 'testing', 'ci/cd', 'pipeline',
            'architecture', 'design', 'framework', 'library', 'dependency'
        }
        
        self.action_indicators = {
            'todo', 'action item', 'next steps', 'follow up', 'assigned to',
            'deadline', 'due date', 'schedule', 'meeting', 'call', 'review',
            'approve', 'implement', 'deploy', 'test', 'fix', 'update', 'create',
            'design', 'build', 'develop', 'investigate', 'research', 'analyze'
        }
        
        # Noise patterns (channel-agnostic)
        self.noise_patterns = [
            r'^(lol|lmao|haha|ok|yes|no|thanks|thx|ty|np|sure|cool|nice)$',
            r'^\w{1,3}$',  # Very short messages
            r'^[^\w]*$',   # Only punctuation/emojis
            r'^(good morning|gm|gn|night|morning|evening)$',
            r'^(brb|afk|gtg|be right back|away from keyboard)$',
            r'^\+1$|^this$|^same$|^agreed$|^exactly$'
        ]
        
        # Channel-specific noise patterns
        self.channel_noise_patterns = {
            'social': [
                r'^(how are you|how\'s it going|what\'s up|wassup)$',
                r'^(have a good|enjoy your|see you).*$',
                r'^(coffee|lunch|break).*$'
            ],
            'general': [
                r'^(anyone|does anyone|has anyone).*\?$',
                r'^(quick question|stupid question).*$'
            ]
        }
    
    async def calculate_relevance_score(self, message: Dict, channel_id: str) -> float:
        """
        Calculate relevance score for a message with channel-specific logic.
        
        Args:
            message: Message data with content, attachments, etc.
            channel_id: Discord channel ID
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        try:
            content = message.get('content', '').strip().lower()
            attachments = message.get('attachments', [])
            
            if not content and not attachments:
                return 0.0
            
            score = 0.0
            
            # Length factor (channel-aware)
            if self._is_alert_channel(channel_id):
                # Short messages OK in alert channels
                score += min(0.4, len(content) / 20)
            elif self._is_social_channel(channel_id):
                # Require longer messages in social channels
                score += min(0.2, len(content) / 100)
            else:
                # Standard length scoring
                score += min(0.3, len(content) / 50)
            
            # Keyword analysis
            if self._has_business_keywords(content):
                score += 0.4
            if self._has_technical_keywords(content):
                score += 0.3
            if self._has_action_indicators(content):
                score += 0.5
            
            # Question/decision indicators
            if '?' in content or any(word in content for word in ['decide', 'decision', 'choose', 'recommend']):
                score += 0.3
            
            # Attachment bonus
            if attachments:
                for attachment in attachments:
                    filename = attachment.lower() if isinstance(attachment, str) else ''
                    if any(ext in filename for ext in ['.pdf', '.docx', '.xlsx', '.pptx']):
                        score += 0.8
                    elif any(ext in filename for ext in ['.png', '.jpg', '.jpeg', '.gif']):
                        score += 0.4
                    else:
                        score += 0.2
            
            # Apply noise penalties
            if self._matches_noise_patterns(content, channel_id):
                score -= 0.5
            
            # Spam/repetitive content penalty
            if self._is_repetitive_content(content):
                score -= 0.3
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.5  # Default neutral score
    
    async def classify_with_cascade(self, content: str, initial_score: float, 
                                   context: Dict = None) -> Dict:
        """
        Two-stage LLM classification: cheap model first, expensive if needed.
        
        Args:
            content: Message content to classify
            initial_score: Pre-computed relevance score
            context: Additional context (channel, user roles, etc.)
            
        Returns:
            Classification result with category, confidence, and reasoning
        """
        try:
            # Stage 1: Fast & Cheap classification
            stage1_result = await self._fast_classify(content, initial_score, context)
            
            # Stage 2: Escalate to powerful model if needed
            should_escalate = (
                stage1_result['confidence'] < 0.65 or
                stage1_result['category'] in ['HIGH', 'CRITICAL']
            )
            
            if should_escalate:
                logger.debug(f"Escalating classification to powerful model (confidence: {stage1_result['confidence']}, category: {stage1_result['category']})")
                stage2_result = await self._precise_classify(content, initial_score, context)
                
                # Use stage 2 result but track both for learning
                final_result = stage2_result
                final_result['escalated'] = True
                final_result['stage1_result'] = stage1_result
            else:
                final_result = stage1_result
                final_result['escalated'] = False
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in cascade classification: {e}")
            return {
                'category': 'MEDIUM',
                'confidence': 0.5,
                'reasoning': f"Classification failed: {str(e)}",
                'escalated': False
            }
    
    async def _fast_classify(self, content: str, initial_score: float, context: Dict = None) -> Dict:
        """Stage 1: Fast classification with gpt-4o-mini."""
        try:
            channel_name = context.get('channel_name', 'unknown') if context else 'unknown'
            user_roles = context.get('user_roles', []) if context else []
            
            prompt = f"""Classify this Discord message for business relevance. Be fast and decisive.

Content: "{content[:500]}"
Channel: {channel_name}
User Roles: {user_roles}
Pre-score: {initial_score:.2f}

Categories:
- CRITICAL: Decisions, policies, financial data, urgent issues
- HIGH: Project updates, strategic discussions, important announcements  
- MEDIUM: Work discussions, business questions, useful information
- LOW: Casual work chat, minor updates, social but work-related
- IRRELEVANT: Personal chat, memes, off-topic, noise

Return JSON: {{"category": "...", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

            response = await llm_client.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a fast, efficient content classifier. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)
            
            # Validate result
            if 'category' not in result or 'confidence' not in result:
                raise ValueError("Invalid response format")
            
            return result
            
        except Exception as e:
            logger.warning(f"Fast classification failed: {e}")
            # Fallback based on initial score
            if initial_score >= 0.7:
                category = 'HIGH'
            elif initial_score >= 0.4:
                category = 'MEDIUM'
            else:
                category = 'LOW'
                
            return {
                'category': category,
                'confidence': 0.5,
                'reasoning': f"Fallback classification (score: {initial_score:.2f})"
            }
    
    async def _precise_classify(self, content: str, initial_score: float, context: Dict = None) -> Dict:
        """Stage 2: Precise classification with powerful model."""
        try:
            channel_name = context.get('channel_name', 'unknown') if context else 'unknown'
            user_roles = context.get('user_roles', []) if context else []
            
            prompt = f"""Perform detailed relevance analysis of this Discord message for business knowledge management.

Content: "{content}"
Channel: {channel_name}
User Roles: {user_roles}
Initial Score: {initial_score:.2f}

Analyze for:
1. Business value and strategic importance
2. Technical significance and implementation details
3. Decision-making content and action items
4. Knowledge preservation value
5. Contextual relevance to channel purpose

Categories (choose most appropriate):
- CRITICAL: Strategic decisions, policies, financial data, critical technical specs, urgent escalations
- HIGH: Project updates, architecture decisions, important announcements, significant discussions
- MEDIUM: General work discussions, technical questions, useful information sharing
- LOW: Casual work chat, minor updates, social but work-related content
- IRRELEVANT: Personal conversations, memes, off-topic content, noise

Provide detailed reasoning for your classification.

Return JSON: {{"category": "...", "confidence": 0.0-1.0, "reasoning": "detailed explanation", "key_topics": ["topic1", "topic2"]}}"""

            response = await llm_client.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert business analyst specializing in knowledge management and content relevance assessment."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)
            
            return result
            
        except Exception as e:
            logger.error(f"Precise classification failed: {e}")
            # Enhanced fallback with reasoning
            if initial_score >= 0.8:
                category = 'CRITICAL'
                reasoning = f"High initial score ({initial_score:.2f}) suggests critical content"
            elif initial_score >= 0.6:
                category = 'HIGH'
                reasoning = f"Good initial score ({initial_score:.2f}) suggests high value"
            elif initial_score >= 0.3:
                category = 'MEDIUM'
                reasoning = f"Moderate initial score ({initial_score:.2f}) suggests medium value"
            else:
                category = 'LOW'
                reasoning = f"Low initial score ({initial_score:.2f}) suggests low value"
                
            return {
                'category': category,
                'confidence': 0.6,
                'reasoning': reasoning,
                'key_topics': []
            }
    
    def _is_alert_channel(self, channel_id: str) -> bool:
        """Check if channel is an alert/monitoring channel."""
        channel_name = channel_id.lower()
        return any(keyword in channel_name for keyword in [
            'alert', 'prod', 'monitor', 'status', 'incident', 'emergency'
        ])
    
    def _is_social_channel(self, channel_id: str) -> bool:
        """Check if channel is primarily social."""
        channel_name = channel_id.lower()
        return any(keyword in channel_name for keyword in [
            'social', 'general', 'random', 'chat', 'lounge', 'coffee'
        ])
    
    def _has_business_keywords(self, content: str) -> bool:
        """Check for business-related keywords."""
        return any(keyword in content for keyword in self.business_keywords)
    
    def _has_technical_keywords(self, content: str) -> bool:
        """Check for technical keywords."""
        return any(keyword in content for keyword in self.technical_keywords)
    
    def _has_action_indicators(self, content: str) -> bool:
        """Check for action items or next steps."""
        return any(indicator in content for indicator in self.action_indicators)
    
    def _matches_noise_patterns(self, content: str, channel_id: str) -> bool:
        """Check if content matches noise patterns."""
        # General noise patterns
        for pattern in self.noise_patterns:
            if re.match(pattern, content, re.IGNORECASE):
                return True
        
        # Channel-specific noise patterns
        channel_type = self._get_channel_type(channel_id)
        if channel_type in self.channel_noise_patterns:
            for pattern in self.channel_noise_patterns[channel_type]:
                if re.match(pattern, content, re.IGNORECASE):
                    return True
        
        return False
    
    def _get_channel_type(self, channel_id: str) -> str:
        """Determine channel type from ID/name."""
        channel_name = channel_id.lower()
        if 'social' in channel_name or 'general' in channel_name:
            return 'social'
        elif 'general' in channel_name:
            return 'general'
        else:
            return 'other'
    
    def _is_repetitive_content(self, content: str) -> bool:
        """Check for repetitive or spam-like content."""
        if len(content) < 10:
            return False
        
        # Check for repeated characters or words
        words = content.split()
        if len(words) > 1:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.5:  # More than 50% repeated words
                return True
        
        # Check for excessive punctuation or caps
        if content.count('!') > 3 or content.count('?') > 3:
            return True
        
        if len([c for c in content if c.isupper()]) / max(1, len(content)) > 0.7:
            return True
        
        return False
    
    async def update_learning_metrics(self, message_id: str, predicted_category: str, 
                                     user_feedback: Optional[bool] = None):
        """Update learning metrics based on user feedback."""
        try:
            # Record the classification for learning
            vita_db.record_relevance_feedback(
                message_id=message_id,
                user_id="system",
                predicted_category=predicted_category,
                actual_usefulness=1 if user_feedback else 0 if user_feedback is False else None
            )
            
            logger.debug(f"Updated learning metrics for message {message_id}")
            
        except Exception as e:
            logger.error(f"Failed to update learning metrics: {e}")

# Global instance
relevance_engine = AdaptiveRelevanceEngine()
