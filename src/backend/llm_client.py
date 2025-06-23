import os
from typing import List, Dict, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv
from .logger import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

class LLMClient:
    """Manages interactions with OpenAI's chat completion API."""
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
    
    async def generate_answer(self, question: str, context_documents: List[Dict], user_id: str = None) -> Dict[str, Any]:
        """
        Generate an answer using the RAG pipeline.
        
        Args:
            question: User's question
            context_documents: List of relevant documents from vector search
            user_id: User ID for logging
            
        Returns:
            Dictionary with answer, confidence, and processing info
        """
        try:
            # Build context from documents
            context_text = self._build_context(context_documents)
            
            # Create system prompt
            system_prompt = self._create_system_prompt()
            
            # Create user prompt with context
            user_prompt = self._create_user_prompt(question, context_text)
            
            logger.info(f"Generating answer for user {user_id} with {len(context_documents)} context documents")
            
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for more consistent answers
                max_tokens=1000,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Calculate confidence based on context relevance
            confidence = self._calculate_confidence(context_documents)
            
            result = {
                "answer": answer,
                "confidence": confidence,
                "context_count": len(context_documents),
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
            logger.info(f"Generated answer with confidence {confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again later.",
                "confidence": 0.0,
                "context_count": 0,
                "tokens_used": 0,
                "error": str(e)
            }
    
    def _build_context(self, context_documents: List[Dict]) -> str:
        """
        Build context text from retrieved documents.
        
        Args:
            context_documents: List of documents with metadata
            
        Returns:
            Formatted context text
        """
        if not context_documents:
            return "No relevant context found."
        
        context_parts = []
        
        for i, doc in enumerate(context_documents, 1):
            metadata = doc.get("metadata", {})
            content = metadata.get("content", "")
            
            # Add document separator and metadata
            context_part = f"--- Document {i} ---\n"
            
            # Add source information if available
            if metadata.get("message_id"):
                context_part += f"Source: Message {metadata.get('message_id')}\n"
            if metadata.get("channel_id"):
                context_part += f"Channel: {metadata.get('channel_id')}\n"
            if metadata.get("timestamp"):
                context_part += f"Timestamp: {metadata.get('timestamp')}\n"
            
            context_part += f"Content: {content}\n\n"
            context_parts.append(context_part)
        
        return "".join(context_parts)
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the LLM."""
        return """You are VITA, an AI knowledge assistant for a Discord community. Your role is to provide helpful, accurate, and contextual answers based on the conversation history and knowledge base.

Guidelines:
1. Answer questions using the provided context documents
2. If the context doesn't contain enough information, say so clearly
3. Be concise but comprehensive in your responses
4. Maintain a helpful and professional tone
5. If you're uncertain about something, express that uncertainty
6. Reference specific messages or sources when relevant
7. Don't make assumptions beyond what's provided in the context

Remember: You are answering based on Discord community conversations and shared knowledge."""
    
    def _create_user_prompt(self, question: str, context: str) -> str:
        """
        Create the user prompt with question and context.
        
        Args:
            question: User's question
            context: Formatted context from documents
            
        Returns:
            Complete user prompt
        """
        return f"""Based on the following context from our Discord community, please answer this question:

Question: {question}

Context:
{context}

Please provide a helpful answer based on the context above. If the context doesn't contain enough information to answer the question completely, please say so."""
    
    def _calculate_confidence(self, context_documents: List[Dict]) -> float:
        """
        Calculate confidence score based on context relevance.
        
        Args:
            context_documents: List of context documents with scores
            
        Returns:
            Confidence score between 0 and 1
        """
        if not context_documents:
            return 0.0
        
        # Get average similarity score
        scores = [doc.get("score", 0.0) for doc in context_documents]
        avg_score = sum(scores) / len(scores)
        
        # Normalize and adjust confidence
        # Pinecone cosine similarity ranges from -1 to 1, but typically 0.7+ is good
        confidence = min(avg_score, 1.0)  # Cap at 1.0
        confidence = max(confidence, 0.0)  # Floor at 0.0
        
        # Apply scaling to make confidence more meaningful
        # Scores above 0.8 are high confidence, below 0.6 are low confidence
        if confidence >= 0.8:
            confidence = 0.8 + (confidence - 0.8) * 2  # Scale 0.8-1.0 to 0.8-1.0
        elif confidence >= 0.6:
            confidence = 0.4 + (confidence - 0.6) * 2  # Scale 0.6-0.8 to 0.4-0.8
        else:
            confidence = confidence * 0.67  # Scale 0.0-0.6 to 0.0-0.4
        
        return min(confidence, 1.0)

# Global LLM client instance
llm_client = LLMClient() 