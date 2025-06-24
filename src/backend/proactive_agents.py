import os
import yaml
from typing import Dict, Any
from .database import vita_db
from .logger import get_logger

logger = get_logger(__name__)

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

# Global instance
ontology_evolution_agent = OntologyEvolutionAgent()
