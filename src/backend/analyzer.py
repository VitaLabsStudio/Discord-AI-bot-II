import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .database import vita_db
from .embedding import embedding_manager
from .llm_client import llm_client
from .ontology import ontology_manager
from .logger import get_logger

logger = get_logger(__name__)

class VitaAnalyzer:
    """Advanced analysis capabilities for VITA including thematic clustering and digests."""
    
    def __init__(self):
        self.min_cluster_size = 3  # Minimum documents per cluster
        self.max_clusters = 8      # Maximum number of thematic clusters
    
    async def generate_thematic_digest(self, days: int = 7) -> Dict[str, Any]:
        """
        Generate a thematic digest for the specified time period.
        
        Args:
            days: Number of days to analyze (default: 7)
            
        Returns:
            Dictionary with digest information
        """
        try:
            logger.info(f"Generating thematic digest for last {days} days")
            
            # Calculate time period
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            # Query Pinecone for vectors in the time period
            documents = await self._get_recent_documents(limit=200)
            
            if len(documents) < self.min_cluster_size:
                logger.info(f"Not enough documents ({len(documents)}) for thematic analysis")
                return {
                    "title": f"Insufficient Data - Last {days} Days",
                    "summary": f"Only {len(documents)} documents found. Need at least {self.min_cluster_size} for thematic analysis.",
                    "themes": [],
                    "document_count": len(documents),
                    "time_period": {"start": start_time.isoformat(), "end": end_time.isoformat()}
                }
            
            # Perform thematic clustering
            clusters = await self._cluster_documents(documents)
            
            # Generate summaries for each cluster
            themes = []
            for i, cluster in enumerate(clusters):
                theme = await self._generate_cluster_summary(cluster, i)
                if theme:
                    themes.append(theme)
            
            # Generate overall digest summary
            overall_summary = await self._generate_overall_summary(themes, len(documents), days)
            
            # Create digest title
            title = f"Thematic Digest - Last {days} Days ({len(themes)} Key Themes)"
            
            # Store the digest
            cluster_info = {
                "num_documents": len(documents),
                "num_clusters": len(clusters),
                "cluster_sizes": [len(cluster) for cluster in clusters]
            }
            
            digest_id = vita_db.save_thematic_digest(
                title=title,
                summary=overall_summary,
                time_period_start=start_time,
                time_period_end=end_time,
                cluster_info=cluster_info
            )
            
            digest = {
                "id": digest_id,
                "title": title,
                "summary": overall_summary,
                "themes": themes,
                "document_count": len(documents),
                "time_period": {"start": start_time.isoformat(), "end": end_time.isoformat()},
                "cluster_info": cluster_info
            }
            
            logger.info(f"Generated thematic digest with {len(themes)} themes")
            return digest
            
        except Exception as e:
            logger.error(f"Failed to generate thematic digest: {e}")
            raise
    
    async def _get_recent_documents(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Get recent documents from Pinecone."""
        try:
            # Create a dummy query to get recent documents
            query_response = embedding_manager.index.query(
                vector=[0.0] * 1536,  # Dummy vector
                top_k=limit,
                include_metadata=True,
                include_values=True
            )
            
            documents = []
            for match in query_response.matches:
                documents.append({
                    'id': match.id,
                    'embedding': match.values,
                    'metadata': match.metadata,
                    'content': match.metadata.get('content', ''),
                    'score': match.score
                })
            
            logger.info(f"Retrieved {len(documents)} recent documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get recent documents: {e}")
            return []
    
    async def _cluster_documents(self, documents: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Cluster documents using KMeans on their embeddings.
        
        Args:
            documents: List of documents with embeddings
            
        Returns:
            List of document clusters
        """
        try:
            if len(documents) < self.min_cluster_size:
                return [documents]  # Return all documents as one cluster
            
            # Extract embeddings
            embeddings = np.array([doc['embedding'] for doc in documents])
            
            # Determine optimal number of clusters
            n_clusters = min(len(documents) // self.min_cluster_size, self.max_clusters)
            n_clusters = max(n_clusters, 1)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Group documents by cluster
            clusters = [[] for _ in range(n_clusters)]
            for doc, label in zip(documents, cluster_labels):
                clusters[label].append(doc)
            
            # Filter out small clusters
            filtered_clusters = [cluster for cluster in clusters if len(cluster) >= self.min_cluster_size]
            
            if not filtered_clusters:
                return [documents]  # Return all documents as one cluster
            
            logger.info(f"Created {len(filtered_clusters)} clusters from {len(documents)} documents")
            return filtered_clusters
            
        except Exception as e:
            logger.error(f"Failed to cluster documents: {e}")
            return [documents]  # Return all documents as one cluster
    
    async def _generate_cluster_summary(self, cluster: List[Dict[str, Any]], cluster_index: int) -> Optional[Dict[str, Any]]:
        """
        Generate a thematic summary for a cluster of documents.
        
        Args:
            cluster: List of documents in the cluster
            cluster_index: Index of the cluster
            
        Returns:
            Theme summary dictionary
        """
        try:
            if not cluster:
                return None
            
            # Combine content from all documents in the cluster
            combined_content = []
            for doc in cluster:
                content = doc.get('content', '').strip()
                if content:
                    combined_content.append(content)
            
            if not combined_content:
                return None
            
            # Create prompt for thematic summarization
            content_text = '\n\n---\n\n'.join(combined_content[:10])  # Limit to first 10 documents
            
            prompt = f"""Analyze the following related messages and identify the main theme.

MESSAGES:
{content_text}

Instructions:
1. Identify the single, overarching theme that connects these messages
2. Summarize the key points, decisions, and outcomes discussed
3. Highlight any important developments or changes
4. Keep the summary concise but informative (2-3 paragraphs max)

Format your response as JSON:
{{
    "theme_title": "Brief descriptive title for this theme",
    "summary": "Detailed summary of the key points and discussions",
    "key_points": ["Point 1", "Point 2", "Point 3"],
    "importance": "high|medium|low"
}}
"""
            
            # Get LLM response
            response = await llm_client.client.chat.completions.create(
                model=llm_client.chat_model,
                messages=[
                    {"role": "system", "content": "You are an expert analyst who creates concise, insightful thematic summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                theme_data = json.loads(response_text)
                theme_data['document_count'] = len(cluster)
                theme_data['cluster_index'] = cluster_index
                
                logger.debug(f"Generated theme summary for cluster {cluster_index}: {theme_data.get('theme_title', 'Unknown')}")
                return theme_data
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response as JSON for cluster {cluster_index}")
                return {
                    "theme_title": f"Theme {cluster_index + 1}",
                    "summary": response_text,
                    "key_points": [],
                    "importance": "medium",
                    "document_count": len(cluster),
                    "cluster_index": cluster_index
                }
                
        except Exception as e:
            logger.error(f"Failed to generate cluster summary for cluster {cluster_index}: {e}")
            return None
    
    async def _generate_overall_summary(self, themes: List[Dict[str, Any]], document_count: int, days: int) -> str:
        """
        Generate an overall summary of all themes.
        
        Args:
            themes: List of theme summaries
            document_count: Total number of documents analyzed
            days: Time period in days
            
        Returns:
            Overall summary text
        """
        try:
            if not themes:
                return f"Analyzed {document_count} documents from the last {days} days but no clear themes emerged."
            
            themes_text = []
            for theme in themes:
                themes_text.append(f"**{theme.get('theme_title', 'Unknown Theme')}** ({theme.get('document_count', 0)} messages): {theme.get('summary', '')}")
            
            prompt = f"""Create a concise executive summary of the key themes and activities from the last {days} days.

IDENTIFIED THEMES:
{chr(10).join(themes_text)}

Instructions:
1. Provide a high-level overview of the main activities and discussions
2. Highlight the most important developments or decisions
3. Note any patterns or trends across themes
4. Keep it concise (1-2 paragraphs)
5. Focus on actionable insights and key takeaways

Write in a professional, executive summary style.
"""
            
            response = await llm_client.client.chat.completions.create(
                model=llm_client.chat_model,
                messages=[
                    {"role": "system", "content": "You are an executive assistant creating high-level summaries for leadership."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate overall summary: {e}")
            return f"Analyzed {document_count} documents from the last {days} days across {len(themes)} key themes."
    
    async def detect_significant_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Detect significant events in the knowledge graph from the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of significant events
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            events = []
            
            # Query for new high-importance nodes
            for concept_name in ['Decision', 'Risk', 'Deadline']:
                nodes = vita_db.query_graph("nodes", label=concept_name)
                for node in nodes:
                    if node.created_at >= cutoff_time:
                        event = {
                            "type": "new_node",
                            "concept": concept_name,
                            "entity": node.name,
                            "timestamp": node.created_at.isoformat(),
                            "importance": "high" if concept_name in ['Decision', 'Risk'] else "medium",
                            "node_id": node.id
                        }
                        events.append(event)
            
            # Query for new relationships involving important concepts
            edges = vita_db.query_graph("edges")
            for edge in edges:
                if edge.created_at >= cutoff_time and edge.relationship in ['manages', 'affects', 'decided_by']:
                    # Get source and target nodes
                    source_nodes = vita_db.query_graph("nodes")
                    target_nodes = vita_db.query_graph("nodes")
                    
                    source_node = next((n for n in source_nodes if n.id == edge.source_id), None)
                    target_node = next((n for n in target_nodes if n.id == edge.target_id), None)
                    
                    if source_node and target_node:
                        event = {
                            "type": "new_relationship",
                            "relationship": edge.relationship,
                            "source": f"{source_node.label}:{source_node.name}",
                            "target": f"{target_node.label}:{target_node.name}",
                            "timestamp": edge.created_at.isoformat(),
                            "importance": "medium",
                            "edge_id": edge.id
                        }
                        events.append(event)
            
            logger.info(f"Detected {len(events)} significant events in last {hours} hours")
            return events
            
        except Exception as e:
            logger.error(f"Failed to detect significant events: {e}")
            return []
    
    async def generate_proactive_briefing(self) -> Optional[Dict[str, Any]]:
        """
        Generate a proactive briefing of important updates.
        
        Returns:
            Briefing dictionary or None if no significant events
        """
        try:
            # Detect significant events from last 24 hours
            events = await self.detect_significant_events(24)
            
            if not events:
                logger.info("No significant events found for proactive briefing")
                return None
            
            # Group events by type and importance
            high_importance = [e for e in events if e.get('importance') == 'high']
            medium_importance = [e for e in events if e.get('importance') == 'medium']
            
            # Generate briefing content
            briefing_sections = []
            
            if high_importance:
                briefing_sections.append(f"🔴 **Critical Updates** ({len(high_importance)} items)")
                for event in high_importance[:5]:  # Limit to top 5
                    if event['type'] == 'new_node':
                        briefing_sections.append(f"• New {event['concept']}: {event['entity']}")
                    elif event['type'] == 'new_relationship':
                        briefing_sections.append(f"• {event['source']} {event['relationship']} {event['target']}")
            
            if medium_importance:
                briefing_sections.append(f"\n🟡 **Notable Updates** ({len(medium_importance)} items)")
                for event in medium_importance[:3]:  # Limit to top 3
                    if event['type'] == 'new_node':
                        briefing_sections.append(f"• New {event['concept']}: {event['entity']}")
                    elif event['type'] == 'new_relationship':
                        briefing_sections.append(f"• {event['source']} {event['relationship']} {event['target']}")
            
            briefing_text = '\n'.join(briefing_sections)
            
            briefing = {
                "title": "VITA Proactive Briefing - Last 24 Hours",
                "content": briefing_text,
                "event_count": len(events),
                "high_priority_count": len(high_importance),
                "medium_priority_count": len(medium_importance),
                "generated_at": datetime.utcnow().isoformat(),
                "events": events
            }
            
            logger.info(f"Generated proactive briefing with {len(events)} events")
            return briefing
            
        except Exception as e:
            logger.error(f"Failed to generate proactive briefing: {e}")
            return None

# Global analyzer instance
vita_analyzer = VitaAnalyzer() 