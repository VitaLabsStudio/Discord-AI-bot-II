import yaml
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from .logger import get_logger

logger = get_logger(__name__)

class OntologyManager:
    """Manages the company ontology for VITA's enhanced understanding."""
    
    def __init__(self, ontology_path: str = "ontology.yaml"):
        self.ontology_path = Path(ontology_path)
        self.ontology = {}
        self.concepts = {}
        self.relationships = {}
        self.notifications = {}
        self._load_ontology()
    
    def _load_ontology(self):
        """Load the ontology from the YAML file."""
        try:
            if not self.ontology_path.exists():
                logger.warning(f"Ontology file not found at {self.ontology_path}")
                self._create_default_ontology()
                return
            
            with open(self.ontology_path, 'r', encoding='utf-8') as f:
                self.ontology = yaml.safe_load(f)
            
            # Parse concepts
            self.concepts = {}
            for concept in self.ontology.get('concepts', []):
                self.concepts[concept['name']] = concept
            
            # Parse relationships
            self.relationships = {}
            for relationship in self.ontology.get('relationships', []):
                self.relationships[relationship['type']] = relationship
            
            # Parse notifications
            self.notifications = self.ontology.get('notifications', {})
            
            logger.info(f"Loaded ontology with {len(self.concepts)} concepts and {len(self.relationships)} relationship types")
            
        except Exception as e:
            logger.error(f"Failed to load ontology: {e}")
            self._create_default_ontology()
    
    def _create_default_ontology(self):
        """Create a minimal default ontology if none exists."""
        default_ontology = {
            'concepts': [
                {
                    'name': 'Project',
                    'description': 'Organized efforts with specific goals',
                    'keywords': ['project', 'initiative', 'effort']
                },
                {
                    'name': 'Person',
                    'description': 'Individuals within the company',
                    'keywords': ['person', 'employee', 'manager']
                }
            ],
            'relationships': [
                {
                    'type': 'manages',
                    'description': 'Person manages Project',
                    'source_concepts': ['Person'],
                    'target_concepts': ['Project']
                }
            ],
            'notifications': {
                'channels': {},
                'roles': {},
                'triggers': []
            }
        }
        
        self.ontology = default_ontology
        self.concepts = {c['name']: c for c in default_ontology['concepts']}
        self.relationships = {r['type']: r for r in default_ontology['relationships']}
        self.notifications = default_ontology['notifications']
        
        logger.info("Created default ontology")
    
    def get_concept_names(self) -> List[str]:
        """Get list of all concept names."""
        return list(self.concepts.keys())
    
    def get_concept(self, name: str) -> Optional[Dict[str, Any]]:
        """Get concept definition by name."""
        return self.concepts.get(name)
    
    def get_relationship_types(self) -> List[str]:
        """Get list of all relationship types."""
        return list(self.relationships.keys())
    
    def get_relationship(self, relationship_type: str) -> Optional[Dict[str, Any]]:
        """Get relationship definition by type."""
        return self.relationships.get(relationship_type)
    
    def create_ontology_prompt(self, content: str) -> str:
        """
        Create a prompt for LLM to tag content with ontology concepts.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Formatted prompt for ontology tagging
        """
        concepts_info = []
        for name, concept in self.concepts.items():
            info = f"- {name}: {concept.get('description', '')}"
            if 'keywords' in concept:
                info += f" (Keywords: {', '.join(concept['keywords'])})"
            if 'examples' in concept:
                info += f" (Examples: {', '.join(concept['examples'][:3])})"
            concepts_info.append(info)
        
        prompt = f"""Given the following text and our company's ontology, identify which concepts from the ontology this text relates to.

COMPANY ONTOLOGY:
{chr(10).join(concepts_info)}

TEXT TO ANALYZE:
{content}

Instructions:
1. Analyze the text for mentions or references to any of the ontology concepts
2. For each concept identified, extract the specific entity name from the text
3. Return your response as a JSON object with this format:
{{
    "concepts": [
        {{"concept": "Project", "entity": "Project Phoenix", "confidence": 0.9}},
        {{"concept": "Person", "entity": "John Doe", "confidence": 0.8}}
    ]
}}

Only include concepts you are confident about (confidence > 0.7). If no concepts are found, return {{"concepts": []}}.
"""
        return prompt
    
    def create_relationship_prompt(self, content: str, entities: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for LLM to extract relationships between entities.
        
        Args:
            content: Text content to analyze
            entities: List of entities identified in the content
            
        Returns:
            Formatted prompt for relationship extraction
        """
        if len(entities) < 2:
            return ""
        
        relationships_info = []
        for rel_type, relationship in self.relationships.items():
            info = f"- {rel_type}: {relationship.get('description', '')}"
            relationships_info.append(info)
        
        entities_list = []
        for entity in entities:
            entities_list.append(f"- {entity['concept']}: {entity['entity']}")
        
        prompt = f"""Given the following text and identified entities, determine what relationships exist between them.

TEXT:
{content}

IDENTIFIED ENTITIES:
{chr(10).join(entities_list)}

POSSIBLE RELATIONSHIPS:
{chr(10).join(relationships_info)}

Instructions:
1. Analyze the text to find explicit or implicit relationships between the entities
2. Only use relationship types from the list above
3. Return your response as a JSON object with this format:
{{
    "relationships": [
        {{"source": "John Doe", "relationship": "manages", "target": "Project Phoenix", "confidence": 0.9}}
    ]
}}

Only include relationships you are confident about (confidence > 0.7). If no relationships are found, return {{"relationships": []}}.
"""
        return prompt
    
    def get_notification_config(self) -> Dict[str, Any]:
        """Get notification configuration."""
        return self.notifications
    
    def should_notify(self, concept: str, importance: str = "normal") -> bool:
        """
        Check if a concept should trigger notifications.
        
        Args:
            concept: Concept name
            importance: Importance level
            
        Returns:
            True if notifications should be sent
        """
        triggers = self.notifications.get('triggers', [])
        for trigger in triggers:
            if trigger.get('concept') == concept:
                trigger_importance = trigger.get('importance', 'normal')
                if importance == 'high' or trigger_importance == importance:
                    return True
        return False
    
    def get_notification_channels(self, concept: str) -> List[str]:
        """
        Get channels that should be notified for a concept.
        
        Args:
            concept: Concept name
            
        Returns:
            List of channel names
        """
        triggers = self.notifications.get('triggers', [])
        for trigger in triggers:
            if trigger.get('concept') == concept:
                return trigger.get('notify_channels', [])
        return []
    
    def get_notification_roles(self, concept: str) -> List[str]:
        """
        Get roles that should be notified for a concept.
        
        Args:
            concept: Concept name
            
        Returns:
            List of role names
        """
        triggers = self.notifications.get('triggers', [])
        for trigger in triggers:
            if trigger.get('concept') == concept:
                return trigger.get('notify_roles', [])
        return []
    
    def reload_ontology(self):
        """Reload the ontology from file."""
        logger.info("Reloading ontology from file")
        self._load_ontology()

# Global ontology manager instance
ontology_manager = OntologyManager() 