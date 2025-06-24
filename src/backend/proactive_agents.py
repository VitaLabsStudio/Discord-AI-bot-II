import os
import yaml
from typing import Dict, Any
from datetime import datetime, timedelta
from .database import vita_db
from .logger import get_logger

logger = get_logger(__name__)

class ChannelMetricUpdateAgent:
    """
    v7.1: Agent that analyzes channel-specific relevance patterns and updates
    dynamic thresholds to optimize filtering accuracy.
    """
    
    def __init__(self):
        self.min_messages_for_analysis = 50  # Minimum messages needed for threshold adjustment
        self.target_filter_rate = 0.3  # Target 30% filtering rate
        self.adjustment_factor = 0.05  # How much to adjust threshold per iteration
        
    async def update_channel_thresholds(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze channel performance and update dynamic thresholds.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Update results and statistics
        """
        try:
            logger.info(f"Updating channel thresholds based on last {days} days")
            
            # Get all channels with activity
            with vita_db.get_session() as session:
                channels = session.query(vita_db.ChannelMetrics).all()
            
            updates_made = []
            total_channels = len(channels)
            
            for channel_metrics in channels:
                channel_id = channel_metrics.channel_id
                current_threshold = channel_metrics.dynamic_relevance_threshold
                total_processed = channel_metrics.total_messages_processed
                irrelevant_count = channel_metrics.irrelevant_messages_count
                
                # Skip channels with insufficient data
                if total_processed < self.min_messages_for_analysis:
                    continue
                
                # Calculate current filter rate
                current_filter_rate = irrelevant_count / total_processed if total_processed > 0 else 0
                
                # Determine if adjustment is needed
                new_threshold = current_threshold
                adjustment_reason = ""
                
                if current_filter_rate > 0.7:  # Too much filtering
                    new_threshold = max(0.1, current_threshold - self.adjustment_factor)
                    adjustment_reason = f"Reducing threshold (filter rate: {current_filter_rate:.2%})"
                elif current_filter_rate < 0.1:  # Too little filtering
                    new_threshold = min(0.8, current_threshold + self.adjustment_factor)
                    adjustment_reason = f"Increasing threshold (filter rate: {current_filter_rate:.2%})"
                
                # Apply update if threshold changed
                if abs(new_threshold - current_threshold) > 0.01:
                    vita_db.update_channel_metrics(channel_id, was_relevant=True)  # Trigger update
                    
                    # Update threshold directly
                    with vita_db.get_session() as session:
                        channel_record = session.query(vita_db.ChannelMetrics).filter(
                            vita_db.ChannelMetrics.channel_id == channel_id
                        ).first()
                        if channel_record:
                            channel_record.dynamic_relevance_threshold = new_threshold
                            session.commit()
                    
                    updates_made.append({
                        "channel_id": channel_id,
                        "old_threshold": current_threshold,
                        "new_threshold": new_threshold,
                        "filter_rate": current_filter_rate,
                        "total_processed": total_processed,
                        "reason": adjustment_reason
                    })
                    
                    logger.info(f"Updated threshold for channel {channel_id}: {current_threshold:.3f} -> {new_threshold:.3f}")
            
            # Get relevance feedback accuracy
            feedback_stats = vita_db.get_relevance_feedback_stats(days)
            
            summary = (
                f"Analyzed {total_channels} channels. "
                f"Updated thresholds for {len(updates_made)} channels. "
                f"Feedback accuracy: {feedback_stats.get('total_feedback', 0)} samples."
            )
            
            return {
                "updates_made": updates_made,
                "total_channels_analyzed": total_channels,
                "channels_updated": len(updates_made),
                "feedback_stats": feedback_stats,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Failed to update channel thresholds: {e}")
            return {"error": str(e), "summary": "Threshold update failed"}
    
    async def analyze_relevance_accuracy(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze the accuracy of relevance classifications.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Accuracy analysis and recommendations
        """
        try:
            feedback_stats = vita_db.get_relevance_feedback_stats(days)
            
            if feedback_stats.get('total_feedback', 0) == 0:
                return {
                    "accuracy_analysis": "No feedback data available",
                    "recommendations": ["Encourage more user feedback to improve accuracy"]
                }
            
            accuracy_by_category = feedback_stats.get('accuracy_by_category', {})
            recommendations = []
            
            # Analyze each category
            for category, stats in accuracy_by_category.items():
                accuracy = stats.get('accuracy', 0)
                total = stats.get('total', 0)
                
                if total >= 5:  # Only analyze categories with sufficient data
                    if accuracy < 0.6:
                        recommendations.append(
                            f"Improve {category} classification (accuracy: {accuracy:.1%})"
                        )
                    elif accuracy > 0.9:
                        recommendations.append(
                            f"{category} classification performing well (accuracy: {accuracy:.1%})"
                        )
            
            if not recommendations:
                recommendations.append("Relevance classification accuracy is acceptable")
            
            return {
                "accuracy_analysis": feedback_stats,
                "recommendations": recommendations,
                "total_feedback_samples": feedback_stats.get('total_feedback', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze relevance accuracy: {e}")
            return {"error": str(e)}

class OntologyEvolutionAgent:
    """
    v6.1: Agent that analyzes failed queries and proposes ontology updates.
    
    This agent identifies knowledge gaps by examining low-confidence or failed
    evidence chains and suggests new concepts to add to the ontology.
    """
    
    def __init__(self, ontology_path: str = "ontology.yaml"):
        self.ontology_path = ontology_path
        self.min_concept_frequency = 3  # Minimum appearances to suggest addition
        self.confidence_threshold = 0.5  # Below this is considered "low confidence"
    
    async def analyze_knowledge_gaps(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze recent failed or low-confidence queries to identify knowledge gaps.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Analysis results with proposed ontology updates
        """
        try:
            logger.info(f"Analyzing knowledge gaps from last {days} days")
            
            # Get failed and low-confidence evidence chains
            failed_chains = vita_db.get_failed_evidence_chains(days)
            
            if not failed_chains:
                logger.info("No failed evidence chains found")
                return {"proposed_concepts": [], "analysis_summary": "No knowledge gaps detected"}
            
            # Extract unknown entities from failed queries
            unknown_entities = await self._extract_unknown_entities(failed_chains)
            
            # Analyze frequency and propose concepts
            proposed_concepts = self._analyze_entity_frequency(unknown_entities)
            
            # Generate ontology diff
            ontology_diff = await self._generate_ontology_diff(proposed_concepts)
            
            analysis_summary = (
                f"Analyzed {len(failed_chains)} failed queries. "
                f"Identified {len(unknown_entities)} unknown entities. "
                f"Proposing {len(proposed_concepts)} new concepts for ontology."
            )
            
            return {
                "proposed_concepts": proposed_concepts,
                "ontology_diff": ontology_diff,
                "analysis_summary": analysis_summary,
                "failed_chains_count": len(failed_chains),
                "unknown_entities_count": len(unknown_entities)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze knowledge gaps: {e}")
            return {"error": str(e), "analysis_summary": "Analysis failed"}

    def _load_current_ontology(self) -> Dict[str, Any]:
        """Load the current ontology YAML file."""
        try:
            if os.path.exists(self.ontology_path):
                with open(self.ontology_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            else:
                logger.warning(f"Ontology file not found at {self.ontology_path}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load ontology file: {e}")
            return {}

# Global instances
channel_metric_agent = ChannelMetricUpdateAgent()
ontology_evolution_agent = OntologyEvolutionAgent()
