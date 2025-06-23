import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import openai
from .logger import get_logger
from .utils import clean_text

logger = get_logger(__name__)

@dataclass
class EntityExtraction:
    """Extracted entity information."""
    text: str
    label: str
    confidence: float
    start_pos: int
    end_pos: int
    context: str  # Surrounding context

@dataclass
class ContentSummary:
    """Content summary information."""
    summary: str
    key_points: List[str]
    topics: List[str]
    word_count: int
    reading_time_minutes: float

@dataclass
class EnrichedContent:
    """Enhanced content with extracted insights."""
    original_content: str
    cleaned_content: str
    summary: Optional[ContentSummary]
    entities: List[EntityExtraction]
    semantic_tags: List[str]
    sentiment_score: float
    language: str
    quality_score: float
    extracted_metadata: Dict[str, Any]

class ContentEnrichmentEngine:
    """Advanced content enrichment with NER, summarization, and semantic analysis."""
    
    def __init__(self):
        self.nlp_model = None
        self.ner_pipeline = None
        self.summarization_pipeline = None
        self.classification_pipeline = None
        self.openai_client = None
        self.initialized = False
        
        # Content quality thresholds
        self.min_content_length = 10
        self.min_quality_score = 0.3
        
        # Entity extraction settings
        self.entity_confidence_threshold = 0.8
        self.context_window = 50  # Characters around entity
        
    async def initialize(self):
        """Initialize all ML models and pipelines."""
        try:
            # Load spaCy model for basic NLP
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, installing...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp_model = spacy.load("en_core_web_sm")
            
            # Initialize Hugging Face pipelines
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=-1  # CPU
            )
            
            self.summarization_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1
            )
            
            self.classification_pipeline = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1
            )
            
            # Initialize OpenAI client
            import os
            if os.getenv("OPENAI_API_KEY"):
                self.openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            self.initialized = True
            logger.info("Content enrichment engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize content enrichment: {e}")
            self.initialized = False
    
    async def enrich_content(self, content: str, content_type: str = "text",
                           enable_summarization: bool = True,
                           enable_ner: bool = True) -> EnrichedContent:
        """
        Perform comprehensive content enrichment.
        
        Args:
            content: Raw content to enrich
            content_type: Type of content (text, document, etc.)
            enable_summarization: Whether to generate summaries
            enable_ner: Whether to extract named entities
            
        Returns:
            EnrichedContent with all extracted insights
        """
        if not self.initialized:
            await self.initialize()
        
        # Clean and validate content
        cleaned_content = clean_text(content)
        quality_score = self._calculate_quality_score(cleaned_content)
        
        if quality_score < self.min_quality_score:
            logger.warning(f"Low quality content detected: score={quality_score}")
        
        # Initialize enriched content
        enriched = EnrichedContent(
            original_content=content,
            cleaned_content=cleaned_content,
            summary=None,
            entities=[],
            semantic_tags=[],
            sentiment_score=0.0,
            language="en",  # Default, could be detected
            quality_score=quality_score,
            extracted_metadata={}
        )
        
        # Run enrichment tasks concurrently
        tasks = []
        
        if enable_ner and len(cleaned_content) > self.min_content_length:
            tasks.append(self._extract_entities(cleaned_content))
        
        if enable_summarization and len(cleaned_content) > 100:
            tasks.append(self._generate_summary(cleaned_content))
        
        tasks.extend([
            self._analyze_sentiment(cleaned_content),
            self._extract_semantic_tags(cleaned_content),
            self._extract_metadata(cleaned_content, content_type)
        ])
        
        # Execute all tasks concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            result_idx = 0
            
            if enable_ner and len(cleaned_content) > self.min_content_length:
                entities_result = results[result_idx]
                if not isinstance(entities_result, Exception):
                    enriched.entities = entities_result
                result_idx += 1
            
            if enable_summarization and len(cleaned_content) > 100:
                summary_result = results[result_idx]
                if not isinstance(summary_result, Exception):
                    enriched.summary = summary_result
                result_idx += 1
            
            # Sentiment analysis
            sentiment_result = results[result_idx]
            if not isinstance(sentiment_result, Exception):
                enriched.sentiment_score = sentiment_result
            result_idx += 1
            
            # Semantic tags
            tags_result = results[result_idx]
            if not isinstance(tags_result, Exception):
                enriched.semantic_tags = tags_result
            result_idx += 1
            
            # Metadata
            metadata_result = results[result_idx]
            if not isinstance(metadata_result, Exception):
                enriched.extracted_metadata = metadata_result
            
        except Exception as e:
            logger.error(f"Error during content enrichment: {e}")
        
        return enriched
    
    async def _extract_entities(self, content: str) -> List[EntityExtraction]:
        """Extract named entities using both spaCy and Hugging Face models."""
        entities = []
        
        try:
            # Use Hugging Face NER pipeline for better accuracy
            ner_results = self.ner_pipeline(content)
            
            for entity in ner_results:
                if entity['score'] >= self.entity_confidence_threshold:
                    # Extract context around entity
                    start = max(0, entity['start'] - self.context_window)
                    end = min(len(content), entity['end'] + self.context_window)
                    context = content[start:end]
                    
                    entities.append(EntityExtraction(
                        text=entity['word'],
                        label=entity['entity_group'],
                        confidence=entity['score'],
                        start_pos=entity['start'],
                        end_pos=entity['end'],
                        context=context
                    ))
            
            # Also use spaCy for additional entity types
            if self.nlp_model:
                doc = self.nlp_model(content)
                for ent in doc.ents:
                    # Skip if already found by Hugging Face model
                    if not any(e.start_pos <= ent.start_char <= e.end_pos for e in entities):
                        context_start = max(0, ent.start_char - self.context_window)
                        context_end = min(len(content), ent.end_char + self.context_window)
                        context = content[context_start:context_end]
                        
                        entities.append(EntityExtraction(
                            text=ent.text,
                            label=ent.label_,
                            confidence=0.9,  # spaCy doesn't provide confidence
                            start_pos=ent.start_char,
                            end_pos=ent.end_char,
                            context=context
                        ))
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
        
        return entities
    
    async def _generate_summary(self, content: str) -> Optional[ContentSummary]:
        """Generate content summary using multiple approaches."""
        try:
            # Calculate reading time
            word_count = len(content.split())
            reading_time = word_count / 200  # Average reading speed
            
            summary_text = ""
            key_points = []
            topics = []
            
            # Use Hugging Face summarization for shorter content
            if len(content) < 1024:
                try:
                    summary_result = self.summarization_pipeline(
                        content,
                        max_length=150,
                        min_length=30,
                        do_sample=False
                    )
                    summary_text = summary_result[0]['summary_text']
                except Exception as e:
                    logger.warning(f"Hugging Face summarization failed: {e}")
            
            # Use OpenAI for better summarization of longer content
            if not summary_text and self.openai_client and len(content) > 100:
                try:
                    response = await self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{
                            "role": "system",
                            "content": "You are a helpful assistant that creates concise summaries and extracts key points from text."
                        }, {
                            "role": "user",
                            "content": f"Please provide:\n1. A concise summary (2-3 sentences)\n2. 3-5 key points\n3. Main topics/themes\n\nText: {content[:2000]}"
                        }],
                        max_tokens=300,
                        temperature=0.3
                    )
                    
                    ai_response = response.choices[0].message.content
                    
                    # Parse the response
                    lines = ai_response.split('\n')
                    current_section = None
                    
                    for line in lines:
                        line = line.strip()
                        if 'summary' in line.lower():
                            current_section = 'summary'
                        elif 'key points' in line.lower():
                            current_section = 'points'
                        elif 'topics' in line.lower() or 'themes' in line.lower():
                            current_section = 'topics'
                        elif line and current_section:
                            if current_section == 'summary' and not summary_text:
                                summary_text = line
                            elif current_section == 'points':
                                key_points.append(line.lstrip('- •'))
                            elif current_section == 'topics':
                                topics.append(line.lstrip('- •'))
                
                except Exception as e:
                    logger.warning(f"OpenAI summarization failed: {e}")
            
            # Fallback: extract first few sentences as summary
            if not summary_text:
                sentences = content.split('. ')[:3]
                summary_text = '. '.join(sentences) + '.'
            
            return ContentSummary(
                summary=summary_text,
                key_points=key_points,
                topics=topics,
                word_count=word_count,
                reading_time_minutes=reading_time
            )
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return None
    
    async def _analyze_sentiment(self, content: str) -> float:
        """Analyze content sentiment."""
        try:
            # Use shorter content for analysis
            analysis_text = content[:512]
            
            result = self.classification_pipeline(analysis_text)
            
            # Convert to numeric score (-1 to 1)
            label = result[0]['label']
            score = result[0]['score']
            
            if 'NEGATIVE' in label.upper():
                return -score
            elif 'POSITIVE' in label.upper():
                return score
            else:
                return 0.0
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return 0.0
    
    async def _extract_semantic_tags(self, content: str) -> List[str]:
        """Extract semantic tags and themes from content."""
        tags = []
        
        try:
            # Use spaCy for basic tagging
            if self.nlp_model:
                doc = self.nlp_model(content)
                
                # Extract noun phrases as potential topics
                for chunk in doc.noun_chunks:
                    if len(chunk.text) > 3 and chunk.text.lower() not in ['this', 'that', 'these', 'those']:
                        tags.append(chunk.text.lower())
                
                # Extract keywords based on POS tags
                keywords = [token.lemma_.lower() for token in doc 
                           if token.pos_ in ['NOUN', 'PROPN'] and 
                           len(token.lemma_) > 2 and
                           not token.is_stop and
                           not token.is_punct]
                
                # Add most frequent keywords
                from collections import Counter
                keyword_counts = Counter(keywords)
                top_keywords = [word for word, count in keyword_counts.most_common(10)]
                tags.extend(top_keywords)
            
            # Remove duplicates and clean
            tags = list(set([tag.strip() for tag in tags if tag.strip()]))
            
            # Limit to top 20 tags
            return tags[:20]
            
        except Exception as e:
            logger.error(f"Semantic tag extraction failed: {e}")
            return []
    
    async def _extract_metadata(self, content: str, content_type: str) -> Dict[str, Any]:
        """Extract structured metadata from content."""
        metadata = {
            "content_type": content_type,
            "character_count": len(content),
            "word_count": len(content.split()),
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
            "line_count": len(content.split('\n')),
            "avg_sentence_length": 0,
            "readability_score": 0,
            "contains_code": False,
            "contains_urls": False,
            "contains_emails": False,
            "language_indicators": []
        }
        
        try:
            # Detect URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, content)
            metadata["contains_urls"] = len(urls) > 0
            metadata["url_count"] = len(urls)
            
            # Detect email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, content)
            metadata["contains_emails"] = len(emails) > 0
            metadata["email_count"] = len(emails)
            
            # Detect code patterns
            code_indicators = ['def ', 'class ', 'import ', 'function ', '```', 'console.log', 'SELECT ', 'FROM ']
            metadata["contains_code"] = any(indicator in content for indicator in code_indicators)
            
            # Calculate average sentence length
            if self.nlp_model:
                doc = self.nlp_model(content[:1000])  # Limit for performance
                sentences = [sent for sent in doc.sents]
                if sentences:
                    avg_length = sum(len(sent.text.split()) for sent in sentences) / len(sentences)
                    metadata["avg_sentence_length"] = avg_length
            
            # Simple readability score (Flesch Reading Ease approximation)
            words = content.split()
            sentences = content.split('.')
            if len(sentences) > 1 and len(words) > 0:
                avg_sentence_length = len(words) / len(sentences)
                # Simplified calculation
                readability = 206.835 - (1.015 * avg_sentence_length)
                metadata["readability_score"] = max(0, min(100, readability))
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
        
        return metadata
    
    def _calculate_quality_score(self, content: str) -> float:
        """Calculate content quality score (0-1)."""
        if not content or len(content) < self.min_content_length:
            return 0.0
        
        score = 0.0
        
        # Length factor (longer content generally better, up to a point)
        length_score = min(1.0, len(content) / 1000)
        score += length_score * 0.3
        
        # Sentence structure (presence of punctuation)
        punctuation_count = sum(1 for char in content if char in '.!?;:')
        word_count = len(content.split())
        if word_count > 0:
            punctuation_ratio = punctuation_count / word_count
            punctuation_score = min(1.0, punctuation_ratio * 10)
            score += punctuation_score * 0.2
        
        # Vocabulary diversity
        words = content.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            diversity_score = len(unique_words) / len(words)
            score += diversity_score * 0.3
        
        # Paragraph structure
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            score += 0.2
        
        return min(1.0, score)
    
    def create_enhanced_metadata(self, enriched: EnrichedContent, 
                                original_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced metadata for Pinecone storage."""
        enhanced_metadata = original_metadata.copy()
        
        # Add enrichment data
        enhanced_metadata.update({
            "quality_score": enriched.quality_score,
            "sentiment_score": enriched.sentiment_score,
            "language": enriched.language,
            "word_count": enriched.extracted_metadata.get("word_count", 0),
            "character_count": enriched.extracted_metadata.get("character_count", 0),
            "contains_code": enriched.extracted_metadata.get("contains_code", False),
            "contains_urls": enriched.extracted_metadata.get("contains_urls", False),
            "readability_score": enriched.extracted_metadata.get("readability_score", 0)
        })
        
        # Add semantic tags (top 10)
        if enriched.semantic_tags:
            enhanced_metadata["semantic_tags"] = enriched.semantic_tags[:10]
        
        # Add entity information
        if enriched.entities:
            entity_types = list(set(e.label for e in enriched.entities))
            enhanced_metadata["entity_types"] = entity_types
            
            # Add high-confidence entities
            high_conf_entities = [e.text for e in enriched.entities if e.confidence > 0.9]
            if high_conf_entities:
                enhanced_metadata["key_entities"] = high_conf_entities[:10]
        
        # Add summary if available
        if enriched.summary:
            enhanced_metadata["has_summary"] = True
            enhanced_metadata["reading_time_minutes"] = enriched.summary.reading_time_minutes
            enhanced_metadata["key_topics"] = enriched.summary.topics[:5]
        
        return enhanced_metadata

# Global enrichment engine
content_enrichment_engine = ContentEnrichmentEngine() 