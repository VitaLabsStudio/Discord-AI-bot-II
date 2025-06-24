"""
VITA v7.1 Viewpoint Matrix System

This module provides multi-dimensional knowledge expansion for critical content:
- Executive summaries for strategic queries
- Technical deep-dives for engineering questions
- Actionable insights for operational planning
- Synthetic Q&A pairs for instant answers
"""

import asyncio
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime, timedelta

from .database import vita_db
from .llm_client import llm_client
from .logger import get_logger
from .utils import clean_text, chunk_content

logger = get_logger(__name__)

class ViewpointMatrix:
    """
    Multi-dimensional content processor that creates specialized viewpoints
    of critical business content for different audiences and use cases.
    """
    
    # Viewpoint configurations with prompts and target audiences
    VIEWPOINTS = {
        'executive': {
            'prompt': """Extract strategic implications, ROI, and key decisions in 3 sentences.
            Focus on: business impact, financial implications, strategic importance, leadership concerns.
            Format: Clear, executive-level summary that a CEO/founder would find valuable.""",
            'weight': 1.5,
            'target_roles': ['executive', 'ceo', 'founder', 'director', 'vp'],
            'max_tokens': 200
        },
        'technical': {
            'prompt': """Extract technical specifications, APIs, architecture details, and engineering risks.
            Focus on: implementation details, technical requirements, system architecture, code references, infrastructure needs.
            Format: Detailed technical analysis for engineers and architects.""",
            'weight': 1.2,
            'target_roles': ['developer', 'engineer', 'architect', 'devops', 'technical'],
            'max_tokens': 400
        },
        'actionable': {
            'prompt': """List all action items, deadlines, and assigned owners in a structured format.
            Focus on: specific tasks, deadlines, responsible parties, next steps, dependencies.
            Format: Bulleted action plan with clear ownership and timelines.""",
            'weight': 1.3,
            'target_roles': ['manager', 'coordinator', 'project manager', 'scrum master'],
            'max_tokens': 300
        },
        'qa_synthetic': {
            'prompt': """Generate 5 Q&A pairs that capture key information from this content.
            Focus on: frequently asked questions, important clarifications, key facts, practical applications.
            Format: Q: [Question] A: [Concise Answer] for each pair.""",
            'weight': 1.0,
            'target_roles': ['all'],
            'max_tokens': 500
        },
        'risk_legal': {
            'prompt': """Identify compliance issues, legal considerations, and risk factors.
            Focus on: regulatory compliance, legal implications, risk assessment, privacy concerns, security issues.
            Format: Risk analysis with severity levels and mitigation recommendations.""",
            'weight': 1.4,
            'target_roles': ['legal', 'compliance', 'security', 'privacy'],
            'max_tokens': 350
        }
    }
    
    def __init__(self):
        self.session_cache = {}  # Cache for cross-document synthesis
    
    async def process_critical_content(self, content: str, metadata: Dict, 
                                     session_id: Optional[str] = None) -> List[Dict]:
        """
        Process critical content through all viewpoint lenses.
        
        Args:
            content: Raw content to analyze
            metadata: Message metadata (channel, user, etc.)
            session_id: Optional session ID for multi-document synthesis
            
        Returns:
            List of viewpoint chunks ready for embedding
        """
        try:
            logger.info(f"Processing critical content through viewpoint matrix: {len(content)} chars")
            
            viewpoint_chunks = []
            
            # Process each viewpoint
            for viewpoint_name, config in self.VIEWPOINTS.items():
                try:
                    analysis = await self._generate_viewpoint_analysis(
                        content, viewpoint_name, config, metadata
                    )
                    
                    if analysis and analysis.strip():
                        # Create chunks for this viewpoint
                        chunks = self._create_viewpoint_chunks(
                            analysis, viewpoint_name, config, metadata
                        )
                        viewpoint_chunks.extend(chunks)
                        
                        # Store in database for retrieval
                        vita_db.store_content_viewpoint(
                            message_id=metadata.get('message_id', 'unknown'),
                            viewpoint_type=viewpoint_name,
                            content=analysis,
                            retrieval_weight=config['weight'],
                            target_roles=config['target_roles']
                        )
                        
                        logger.debug(f"Generated {viewpoint_name} viewpoint: {len(analysis)} chars")
                    
                except Exception as e:
                    logger.error(f"Failed to generate {viewpoint_name} viewpoint: {e}")
                    continue
            
            # Handle cross-document synthesis if session_id provided
            if session_id:
                await self._handle_session_synthesis(content, metadata, session_id)
            
            logger.info(f"Generated {len(viewpoint_chunks)} viewpoint chunks")
            return viewpoint_chunks
            
        except Exception as e:
            logger.error(f"Failed to process critical content through viewpoint matrix: {e}")
            return []
    
    async def _generate_viewpoint_analysis(self, content: str, viewpoint_name: str, 
                                         config: Dict, metadata: Dict) -> str:
        """Generate analysis for a specific viewpoint."""
        try:
            channel_context = metadata.get('channel_id', 'unknown')
            user_roles = metadata.get('user_roles', [])
            
            system_prompt = f"""You are a specialized analyst creating {viewpoint_name} insights for business knowledge management.
Your analysis will be used to answer questions from users with these roles: {config['target_roles']}.
Channel context: {channel_context}
User roles: {user_roles}"""

            user_prompt = f"""Analyze this content from a {viewpoint_name} perspective:

CONTENT:
{content}

ANALYSIS INSTRUCTIONS:
{config['prompt']}

REQUIREMENTS:
- Be specific and actionable
- Focus on the {viewpoint_name} perspective
- Extract only the most valuable insights
- Keep within {config['max_tokens']} tokens
- Use clear, professional language"""

            response = await llm_client.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=config['max_tokens']
            )
            
            analysis = response.choices[0].message.content.strip()
            return clean_text(analysis)
            
        except Exception as e:
            logger.error(f"Failed to generate {viewpoint_name} analysis: {e}")
            return ""
    
    def _create_viewpoint_chunks(self, analysis: str, viewpoint_name: str, 
                               config: Dict, metadata: Dict) -> List[Dict]:
        """Create embedding-ready chunks from viewpoint analysis."""
        try:
            # For Q&A viewpoints, split into individual Q&A pairs
            if viewpoint_name == 'qa_synthetic':
                return self._create_qa_chunks(analysis, config, metadata)
            
            # For other viewpoints, create semantic chunks
            base_chunks = chunk_content(analysis, max_chunk_size=300, overlap=50)
            
            viewpoint_chunks = []
            for i, chunk in enumerate(base_chunks):
                chunk_data = {
                    'content': chunk,
                    'metadata': {
                        **metadata,
                        'viewpoint': viewpoint_name,
                        'retrieval_weight': config['weight'],
                        'target_roles': config['target_roles'],
                        'chunk_index': i,
                        'total_chunks': len(base_chunks),
                        'content_type': 'viewpoint_analysis'
                    }
                }
                viewpoint_chunks.append(chunk_data)
            
            return viewpoint_chunks
            
        except Exception as e:
            logger.error(f"Failed to create viewpoint chunks: {e}")
            return []
    
    def _create_qa_chunks(self, qa_analysis: str, config: Dict, metadata: Dict) -> List[Dict]:
        """Create individual chunks for Q&A pairs."""
        try:
            qa_chunks = []
            
            # Parse Q&A pairs from the analysis
            qa_pairs = self._parse_qa_pairs(qa_analysis)
            
            for i, (question, answer) in enumerate(qa_pairs):
                # Create a chunk for each Q&A pair
                qa_content = f"Q: {question}\nA: {answer}"
                
                chunk_data = {
                    'content': qa_content,
                    'metadata': {
                        **metadata,
                        'viewpoint': 'qa_synthetic',
                        'retrieval_weight': config['weight'],
                        'target_roles': config['target_roles'],
                        'qa_index': i,
                        'question': question,
                        'answer': answer,
                        'content_type': 'synthetic_qa'
                    }
                }
                qa_chunks.append(chunk_data)
            
            return qa_chunks
            
        except Exception as e:
            logger.error(f"Failed to create Q&A chunks: {e}")
            return []
    
    def _parse_qa_pairs(self, qa_text: str) -> List[Tuple[str, str]]:
        """Parse Q&A pairs from generated text."""
        try:
            pairs = []
            lines = qa_text.split('\n')
            current_q = None
            current_a = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('Q:') or line.startswith('Question:'):
                    if current_q and current_a:
                        pairs.append((current_q, current_a))
                    current_q = line[2:].strip() if line.startswith('Q:') else line[9:].strip()
                    current_a = None
                elif line.startswith('A:') or line.startswith('Answer:'):
                    current_a = line[2:].strip() if line.startswith('A:') else line[7:].strip()
                elif current_a and line:
                    current_a += ' ' + line
            
            # Add the last pair
            if current_q and current_a:
                pairs.append((current_q, current_a))
            
            return pairs
            
        except Exception as e:
            logger.error(f"Failed to parse Q&A pairs: {e}")
            return []
    
    async def _handle_session_synthesis(self, content: str, metadata: Dict, session_id: str):
        """Handle cross-document synthesis for multi-file uploads."""
        try:
            # Store content for this session
            if session_id not in self.session_cache:
                self.session_cache[session_id] = {
                    'documents': [],
                    'created_at': datetime.utcnow()
                }
            
            self.session_cache[session_id]['documents'].append({
                'content': content,
                'metadata': metadata,
                'filename': metadata.get('filename', 'unknown')
            })
            
            # If we have multiple documents, generate synthesis
            session_data = self.session_cache[session_id]
            if len(session_data['documents']) >= 2:
                await self._generate_cross_document_synthesis(session_id, session_data)
            
        except Exception as e:
            logger.error(f"Failed to handle session synthesis: {e}")
    
    async def _generate_cross_document_synthesis(self, session_id: str, session_data: Dict):
        """Generate synthesis across multiple documents in a session."""
        try:
            documents = session_data['documents']
            
            # Create combined content summary
            doc_summaries = []
            for i, doc in enumerate(documents):
                filename = doc['metadata'].get('filename', f'Document {i+1}')
                content_preview = doc['content'][:500] + '...' if len(doc['content']) > 500 else doc['content']
                doc_summaries.append(f"Document: {filename}\nContent: {content_preview}")
            
            combined_summary = '\n\n'.join(doc_summaries)
            
            synthesis_prompt = f"""Analyze these related documents uploaded together and create a comprehensive synthesis.

DOCUMENTS:
{combined_summary}

Generate a synthesis that:
1. Identifies common themes and connections between documents
2. Highlights complementary information and relationships
3. Notes any conflicts or discrepancies
4. Provides strategic insights from the combined information
5. Suggests how these documents work together

Keep the synthesis comprehensive but focused on the most valuable cross-document insights."""

            response = await llm_client.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a strategic analyst specializing in document synthesis and relationship identification."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            
            synthesis = response.choices[0].message.content.strip()
            
            # Store synthesis as a special viewpoint
            synthesis_metadata = {
                'session_id': session_id,
                'document_count': len(documents),
                'synthesis_type': 'cross_document',
                'content_type': 'document_synthesis'
            }
            
            vita_db.store_content_viewpoint(
                message_id=f"synthesis_{session_id}",
                viewpoint_type='cross_document_synthesis',
                content=synthesis,
                retrieval_weight=2.0,  # High weight for synthesis
                target_roles=['all']
            )
            
            logger.info(f"Generated cross-document synthesis for session {session_id}: {len(synthesis)} chars")
            
        except Exception as e:
            logger.error(f"Failed to generate cross-document synthesis: {e}")
    
    async def get_role_specific_content(self, message_id: str, user_roles: List[str]) -> List[Dict]:
        """Get viewpoint content tailored to specific user roles."""
        try:
            # Get all viewpoints for the message
            viewpoints = vita_db.get_viewpoints_for_message(message_id)
            
            # Filter and rank by role relevance
            relevant_viewpoints = []
            for viewpoint in viewpoints:
                target_roles = viewpoint.get('target_roles', [])
                
                # Check if any user role matches target roles
                if 'all' in target_roles or any(role.lower() in [tr.lower() for tr in target_roles] for role in user_roles):
                    # Calculate role match score
                    role_match_score = self._calculate_role_match_score(user_roles, target_roles)
                    viewpoint['role_match_score'] = role_match_score
                    relevant_viewpoints.append(viewpoint)
            
            # Sort by retrieval weight and role match
            relevant_viewpoints.sort(
                key=lambda x: (x['retrieval_weight'] * x['role_match_score']), 
                reverse=True
            )
            
            return relevant_viewpoints
            
        except Exception as e:
            logger.error(f"Failed to get role-specific content: {e}")
            return []
    
    def _calculate_role_match_score(self, user_roles: List[str], target_roles: List[str]) -> float:
        """Calculate how well user roles match target roles."""
        if 'all' in target_roles:
            return 1.0
        
        if not user_roles or not target_roles:
            return 0.5
        
        user_roles_lower = [role.lower() for role in user_roles]
        target_roles_lower = [role.lower() for role in target_roles]
        
        # Count exact matches
        exact_matches = sum(1 for role in user_roles_lower if role in target_roles_lower)
        
        # Count partial matches (substring matching)
        partial_matches = 0
        for user_role in user_roles_lower:
            for target_role in target_roles_lower:
                if user_role in target_role or target_role in user_role:
                    partial_matches += 0.5
        
        total_matches = exact_matches + partial_matches
        max_possible = max(len(user_roles), len(target_roles))
        
        return min(1.0, total_matches / max_possible)
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old session data from memory."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            sessions_to_remove = []
            for session_id, session_data in self.session_cache.items():
                if session_data['created_at'] < cutoff_time:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.session_cache[session_id]
            
            if sessions_to_remove:
                logger.info(f"Cleaned up {len(sessions_to_remove)} old synthesis sessions")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")

# Global instance
viewpoint_processor = ViewpointMatrix()
