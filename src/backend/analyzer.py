import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sklearn.cluster import KMeans
import numpy as np
from .database import vita_db
from .embedding import embedding_manager
from .llm_client import llm_client
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

    # v5.1: Knowledge Lifecycle Management Methods
    
    async def detect_knowledge_supersession(self, new_message_content: str, message_id: str) -> List[Dict[str, Any]]:
        """
        Detect if new content supersedes existing knowledge nodes.
        
        Args:
            new_message_content: Content of the new message
            message_id: ID of the message
            
        Returns:
            List of supersession actions taken
        """
        try:
            actions = []
            
            # Use LLM to detect potential supersessions  
            supersession_prompt = f"""Analyze this new message content to identify if it supersedes or updates any existing decisions, policies, or procedures.

New Content: {new_message_content}

Instructions:
1. Look for explicit language like "updated", "changed", "replaced", "supersedes", "new version"
2. Identify what specific decision, policy, or procedure is being updated
3. Determine if this is a complete replacement or partial update

Format your response as JSON:
{{
    "has_supersession": true|false,
    "superseded_items": [
        {{
            "type": "Decision|Policy|Procedure|Project",
            "name": "Name of the item being superseded",
            "confidence": 0.0-1.0,
            "reason": "Why this is considered superseded"
        }}
    ]
}}
"""
            
            response = await llm_client.client.chat.completions.create(
                model=llm_client.chat_model,
                messages=[
                    {"role": "system", "content": "You are an expert at detecting when new information supersedes existing knowledge."},
                    {"role": "user", "content": supersession_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            import json
            result = json.loads(response.choices[0].message.content.strip())
            
            if result.get("has_supersession", False):
                for item in result.get("superseded_items", []):
                    # Find existing nodes that match
                    existing_nodes = vita_db.get_active_nodes(
                        label=item["type"], 
                        name=item["name"]
                    )
                    
                    for old_node in existing_nodes:
                        # Create new node for the updated information
                        new_node_id = vita_db.create_or_get_node(
                            label=item["type"],
                            name=f"{item['name']} (Updated)",
                            metadata={"updated_from_message": message_id, "confidence": item["confidence"]}
                        )
                        
                        # Supersede the old node
                        success = vita_db.supersede_node(old_node.id, new_node_id, message_id)
                        
                        if success:
                            actions.append({
                                "action": "superseded",
                                "old_node_id": old_node.id,
                                "new_node_id": new_node_id,
                                "item_type": item["type"],
                                "item_name": item["name"],
                                "confidence": item["confidence"],
                                "reason": item["reason"]
                            })
                            
                            logger.info(f"Superseded {item['type']} '{item['name']}' with confidence {item['confidence']}")
            
            return actions
            
        except Exception as e:
            logger.error(f"Failed to detect knowledge supersession: {e}")
            return []
    
    async def review_playbook_performance(self) -> Dict[str, Any]:
        """
        Review playbook and SOP performance based on user feedback.
        
        Returns:
            Review summary with recommendations
        """
        try:
            # Get playbook feedback summary from last 30 days
            playbook_stats = vita_db.get_playbook_feedback_summary(30)
            
            flagged_playbooks = []
            recommendations = []
            
            for playbook in playbook_stats:
                total_usage = playbook['total_usage']
                negative_feedback = playbook['negative_feedback']
                
                if total_usage > 0:
                    negative_rate = negative_feedback / total_usage
                    
                    # Flag playbooks with >50% negative feedback and at least 3 uses
                    if negative_rate > 0.5 and total_usage >= 3:
                        flagged_playbooks.append({
                            **playbook,
                            "negative_rate": negative_rate,
                            "status": "needs_review"
                        })
                        
                        recommendations.append({
                            "type": "playbook_review",
                            "playbook_name": playbook['name'],
                            "issue": f"High negative feedback rate: {negative_rate:.1%}",
                            "suggestion": "Review and update this playbook based on user feedback"
                        })
                    
                    # Flag unused playbooks (no usage in 30 days) if they reference superseded nodes
                    elif total_usage == 0:
                        # Check if playbook references superseded nodes
                        superseded_nodes = vita_db.get_superseded_nodes(30)
                        for superseded_info in superseded_nodes:
                            superseded_node = superseded_info[0]  # First element is the superseded node
                            if superseded_node.id == playbook['node_id']:
                                flagged_playbooks.append({
                                    **playbook,
                                    "status": "references_superseded"
                                })
                                
                                recommendations.append({
                                    "type": "playbook_obsolete",
                                    "playbook_name": playbook['name'],
                                    "issue": "References superseded knowledge",
                                    "suggestion": "Archive or update this playbook"
                                })
            
            review_summary = {
                "review_date": datetime.utcnow().isoformat(),
                "total_playbooks_reviewed": len(playbook_stats),
                "flagged_playbooks": len(flagged_playbooks),
                "recommendations": recommendations,
                "flagged_details": flagged_playbooks
            }
            
            logger.info(f"Playbook review completed: {len(flagged_playbooks)} playbooks flagged")
            return review_summary
            
        except Exception as e:
            logger.error(f"Failed to review playbook performance: {e}")
            return {}

    # v5.1: Predictive Intelligence & Strategic Advisory Methods
    
    async def detect_downstream_risks(self, source_risk_node_id: int) -> List[Dict[str, Any]]:
        """
        Detect downstream risks by traversing dependency relationships in the knowledge graph.
        
        Args:
            source_risk_node_id: ID of the node where risk was detected
            
        Returns:
            List of downstream impact alerts
        """
        try:
            alerts = []
            
            # Get relationships where the risk node is the source of dependencies
            edges = vita_db.query_graph("edges", source_id=source_risk_node_id)
            
            # Also get relationships where other nodes depend on this risk node
            dependent_edges = vita_db.query_graph("edges", target_id=source_risk_node_id)
            
            # Process dependencies
            all_edges = edges + [edge for edge in dependent_edges if edge.relationship == "depends_on"]
            
            for edge in all_edges:
                if edge.relationship in ["depends_on", "affects", "impacts"]:
                    # Get the dependent node
                    dependent_node_id = edge.target_id if edge.source_id == source_risk_node_id else edge.source_id
                    dependent_nodes = [node for node in vita_db.query_graph("nodes") if node.id == dependent_node_id]
                    
                    if dependent_nodes:
                        dependent_node = dependent_nodes[0]
                        
                        # Generate contextualized alert
                        alert = await self._generate_downstream_alert(
                            source_risk_node_id, dependent_node, edge.relationship
                        )
                        
                        if alert:
                            alerts.append(alert)
            
            logger.info(f"Generated {len(alerts)} downstream risk alerts")
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to detect downstream risks: {e}")
            return []
    
    async def _generate_downstream_alert(self, source_node_id: int, dependent_node: Any, 
                                       relationship: str) -> Optional[Dict[str, Any]]:
        """
        Generate a contextualized downstream alert.
        
        Args:
            source_node_id: Source risk node ID
            dependent_node: Dependent node object
            relationship: Type of relationship
            
        Returns:
            Alert dictionary
        """
        try:
            # Get source node details
            source_nodes = [node for node in vita_db.query_graph("nodes") if node.id == source_node_id]
            if not source_nodes:
                return None
            
            source_node = source_nodes[0]
            
            # Generate alert message
            alert_prompt = f"""Generate a concise downstream risk alert.

Source Issue: {source_node.label} "{source_node.name}" has detected risks
Impacted Entity: {dependent_node.label} "{dependent_node.name}"
Relationship: {relationship}

Create a brief, professional alert message that:
1. Explains the potential downstream impact
2. Suggests what the impacted team should monitor
3. Provides context without revealing sensitive details

Format as a single paragraph, under 150 words.
"""
            
            response = await llm_client.client.chat.completions.create(
                model=llm_client.chat_model,
                messages=[
                    {"role": "system", "content": "You are a strategic risk analyst who creates clear, actionable alerts."},
                    {"role": "user", "content": alert_prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            
            alert_message = response.choices[0].message.content.strip()
            
            return {
                "type": "downstream_risk",
                "source_node_id": source_node_id,
                "source_entity": f"{source_node.label}:{source_node.name}",
                "impacted_node_id": dependent_node.id,
                "impacted_entity": f"{dependent_node.label}:{dependent_node.name}",
                "relationship": relationship,
                "alert_message": alert_message,
                "priority": "medium",
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate downstream alert: {e}")
            return None
    
    async def generate_leadership_digest(self) -> Optional[Dict[str, Any]]:
        """
        Generate a comprehensive weekly leadership digest.
        
        Returns:
            Leadership digest with key decisions, risks, and knowledge gaps
        """
        try:
            logger.info("Generating weekly leadership digest")
            
            # Gather signals from multiple sources
            signals = await self._gather_leadership_signals()
            
            if not any(signals.values()):
                logger.info("No significant signals found for leadership digest")
                return None
            
            # Generate structured digest using LLM
            digest_content = await self._synthesize_leadership_digest(signals)
            
            digest = {
                "title": "VITA Leadership Digest - Weekly Summary",
                "generated_at": datetime.utcnow().isoformat(),
                "time_period": f"Week of {(datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')}",
                "content": digest_content,
                "signals_summary": {
                    "new_decisions": len(signals["new_decisions"]),
                    "new_risks": len(signals["new_risks"]),
                    "knowledge_gaps": len(signals["knowledge_gaps"]),
                    "flagged_playbooks": len(signals["flagged_playbooks"])
                },
                "raw_signals": signals
            }
            
            logger.info("Generated leadership digest successfully")
            return digest
            
        except Exception as e:
            logger.error(f"Failed to generate leadership digest: {e}")
            return None
    
    async def _gather_leadership_signals(self) -> Dict[str, List]:
        """
        Gather signals from across the VITA system for leadership digest.
        
        Returns:
            Dictionary of categorized signals
        """
        try:
            signals = {
                "new_decisions": [],
                "new_risks": [],
                "knowledge_gaps": [],
                "flagged_playbooks": []
            }
            
            # Get new decision and risk nodes from the last week
            recent_nodes = vita_db.query_graph("nodes")
            week_ago = datetime.utcnow() - timedelta(days=7)
            
            for node in recent_nodes:
                if node.created_at >= week_ago:
                    if node.label == "Decision":
                        signals["new_decisions"].append({
                            "name": node.name,
                            "created_at": node.created_at.isoformat(),
                            "properties": node.properties
                        })
                    elif node.label == "Risk":
                        signals["new_risks"].append({
                            "name": node.name,
                            "created_at": node.created_at.isoformat(),
                            "properties": node.properties
                        })
            
            # Get failed evidence chains (knowledge gaps)
            failed_chains = vita_db.get_failed_evidence_chains(7)
            for chain in failed_chains:
                signals["knowledge_gaps"].append({
                    "query": chain.user_query,
                    "timestamp": chain.timestamp.isoformat(),
                    "reasoning_plan": chain.reasoning_plan
                })
            
            # Get flagged playbooks from performance review
            playbook_review = await self.review_playbook_performance()
            signals["flagged_playbooks"] = playbook_review.get("flagged_details", [])
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to gather leadership signals: {e}")
            return {"new_decisions": [], "new_risks": [], "knowledge_gaps": [], "flagged_playbooks": []}
    
    async def _synthesize_leadership_digest(self, signals: Dict[str, List]) -> str:
        """
        Synthesize leadership digest content using LLM.
        
        Args:
            signals: Dictionary of gathered signals
            
        Returns:
            Formatted digest content
        """
        try:
            signals_text = f"""New Decisions ({len(signals['new_decisions'])} items):
{chr(10).join([f"• {d['name']}" for d in signals['new_decisions'][:5]])}

New Risks & Blockers ({len(signals['new_risks'])} items):
{chr(10).join([f"• {r['name']}" for r in signals['new_risks'][:5]])}

Knowledge Gaps ({len(signals['knowledge_gaps'])} items):
{chr(10).join([f"• {g['query'][:100]}..." for g in signals['knowledge_gaps'][:3]])}

Playbooks Needing Attention ({len(signals['flagged_playbooks'])} items):
{chr(10).join([f"• {p['name']} - {p.get('status', 'unknown')}" for p in signals['flagged_playbooks'][:3]])}
"""
            
            digest_prompt = f"""You are an AI Chief of Staff creating a weekly leadership digest. Based on the following data points from the past week, generate a concise executive summary.

DATA POINTS:
{signals_text}

Instructions:
1. Create three main sections: "Key Decisions Made", "New Risks & Blockers", "Knowledge That Needs Attention"
2. Be brief, factual, and strategic
3. Focus on actionable insights and implications
4. Use executive-appropriate language
5. Highlight patterns or trends if visible
6. Maximum 400 words total

Format with clear section headers and bullet points.
"""
            
            response = await llm_client.client.chat.completions.create(
                model=llm_client.chat_model,
                messages=[
                    {"role": "system", "content": "You are an expert executive assistant who creates insightful, concise leadership summaries."},
                    {"role": "user", "content": digest_prompt}
                ],
                temperature=0.1,
                max_tokens=600
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to synthesize leadership digest: {e}")
            return "Unable to generate digest content due to processing error."

# Global analyzer instance
vita_analyzer = VitaAnalyzer() 