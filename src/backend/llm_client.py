import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
from cachetools import TTLCache
from .database import vita_db
from .logger import get_logger
from .retry_utils import retry_on_api_error

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

class LLMClient:
    """Enhanced LLM client with conversational memory, role adaptation, and v6.1 observability."""
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
        
        # Conversational memory cache (TTL = 1 hour)
        self.conversation_cache = TTLCache(maxsize=1000, ttl=3600)
        
        # v6.1: Track usage for observability
        self.track_usage = True
    
    def _track_llm_usage(self, model: str, tokens: int, operation: str):
        """Track LLM usage if observability is enabled."""
        if self.track_usage:
            try:
                # Import here to avoid circular imports
                from .api import track_llm_usage
                track_llm_usage(model, tokens, operation)
            except ImportError:
                # Fallback logging if metrics not available
                logger.debug(f"LLM usage: {model} used {tokens} tokens for {operation}")
    
    @retry_on_api_error(max_attempts=3, min_wait=1.0, max_wait=30.0)
    async def generate_answer(self, question: str, context_documents: List[Dict], 
                            user_id: str = None, user_roles: List[str] = None) -> Dict[str, Any]:
        """
        Generate an answer using the RAG pipeline with retry logic, conversational memory, and role adaptation.
        
        v6.1: Enhanced with token usage tracking for observability.
        
        Args:
            question: User's question
            context_documents: List of relevant documents from vector search
            user_id: User ID for logging and conversation tracking
            user_roles: User's roles for response adaptation
            
        Returns:
            Dictionary with answer, confidence, and processing info
        """
        try:
            # Get conversation history for context
            conversation_history = self._get_conversation_history(user_id) if user_id else []
            
            # Build context from documents
            context_text = self._build_context(context_documents)
            
            # Create role-adaptive system prompt
            system_prompt = self._create_system_prompt(user_roles, conversation_history)
            
            # Create user prompt with context and conversation history
            user_prompt = self._create_user_prompt(question, context_text, conversation_history)
            
            logger.info(f"Generating answer for user {user_id} with {len(context_documents)} context documents and {len(conversation_history)} conversation history")
            
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
            
            # v6.1: Track token usage
            if response.usage:
                total_tokens = response.usage.total_tokens
                self._track_llm_usage(self.chat_model, total_tokens, "generate_answer")
            
            # Calculate confidence based on context relevance
            confidence = self._calculate_confidence(context_documents)
            
            # v6.1: Track confidence metrics
            try:
                from .api import vita_query_confidence
                vita_query_confidence.observe(confidence)
            except ImportError:
                pass
            
            # Save conversation to memory
            if user_id:
                self._save_conversation(user_id, question, answer)
            
            result = {
                "answer": answer,
                "confidence": confidence,
                "context_count": len(context_documents),
                "conversation_context": len(conversation_history),
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
    
    def _create_system_prompt(self, user_roles: Optional[List[str]] = None, 
                             conversation_history: Optional[List[Dict]] = None) -> str:
        """Create a role-adaptive system prompt for the LLM."""
        
        base_prompt = """You are VITA, an AI knowledge assistant for a Discord community. Your role is to provide helpful, accurate, and contextual answers based on the conversation history and knowledge base."""
        
        # Role-specific adaptations
        role_adaptations = []
        if user_roles:
            for role in user_roles:
                if any(exec_role in role.lower() for exec_role in ['ceo', 'cto', 'vp', 'director', 'executive']):
                    role_adaptations.append("You are speaking to an executive. Provide high-level, strategic summaries first. Focus on outcomes, decisions, and business impact. Be concise and emphasize actionable insights.")
                elif any(mgmt_role in role.lower() for mgmt_role in ['manager', 'lead', 'project manager']):
                    role_adaptations.append("You are speaking to a manager. Include operational details and team-relevant information. Highlight project status, blockers, and coordination needs.")
                elif any(tech_role in role.lower() for tech_role in ['engineer', 'developer', 'technical']):
                    role_adaptations.append("You are speaking to an engineer. Provide technical details, include code snippets if relevant, and cite specific technical documents. Be precise about implementation details.")
        
        # Conversation context
        context_guidance = ""
        if conversation_history:
            context_guidance = "\n\nConversation Context: You have access to the recent conversation history with this user. Use it to provide contextual follow-up answers and avoid repeating information."
        
        # Combine all parts
        full_prompt = base_prompt
        if role_adaptations:
            full_prompt += "\n\nRole-Specific Guidance:\n" + "\n".join(role_adaptations)
        
        full_prompt += context_guidance
        
        full_prompt += """

Core Guidelines:
1. Answer questions using the provided context documents
2. If the context doesn't contain enough information, say so clearly
3. Maintain a helpful and professional tone appropriate to the user's role
4. If you're uncertain about something, express that uncertainty
5. Reference specific messages or sources when relevant
6. Don't make assumptions beyond what's provided in the context
7. Use conversation history to provide better follow-up responses

Remember: You are answering based on Discord community conversations and shared knowledge."""
        
        return full_prompt
    
    def _create_user_prompt(self, question: str, context: str, 
                           conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Create the user prompt with question, context, and conversation history.
        
        Args:
            question: User's question
            context: Formatted context from documents
            conversation_history: Recent conversation exchanges
            
        Returns:
            Complete user prompt
        """
        prompt_parts = []
        
        # Add conversation history if available
        if conversation_history:
            prompt_parts.append("Recent Conversation History:")
            for i, exchange in enumerate(conversation_history[-3:], 1):  # Last 3 exchanges
                prompt_parts.append(f"Exchange {i}:")
                prompt_parts.append(f"Q: {exchange.get('query_text', '')}")
                prompt_parts.append(f"A: {exchange.get('answer_text', '')}")
                prompt_parts.append("")
        
        prompt_parts.extend([
            "Based on the following context from our Discord community, please answer this question:",
            "",
            f"Question: {question}",
            "",
            "Context:",
            context,
            "",
            "Please provide a helpful answer based on the context above. If the context doesn't contain enough information to answer the question completely, please say so."
        ])
        
        if conversation_history:
            prompt_parts.append("Consider the conversation history to provide a contextual response that builds on previous exchanges.")
        
        return "\n".join(prompt_parts)
    
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
    
    def _get_conversation_history(self, user_id: str) -> List[Dict]:
        """
        Get conversation history for a user from cache or database.
        
        Args:
            user_id: User ID
            
        Returns:
            List of recent conversation exchanges
        """
        # Check cache first
        if user_id in self.conversation_cache:
            return self.conversation_cache[user_id]
        
        # Get from database
        conversations = vita_db.get_conversation_history(user_id, limit=5)
        
        # Convert to format expected by prompt
        history = []
        for conv in conversations:
            history.append({
                'query_text': conv.query_text,
                'answer_text': conv.answer_text,
                'timestamp': conv.timestamp.isoformat()
            })
        
        # Cache for future use
        self.conversation_cache[user_id] = history
        
        return history
    
    def _save_conversation(self, user_id: str, question: str, answer: str):
        """
        Save conversation exchange to memory and database.
        
        Args:
            user_id: User ID
            question: User's question
            answer: VITA's response
        """
        try:
            # Save to database
            vita_db.save_conversation(user_id, question, answer)
            
            # Update cache
            history = self._get_conversation_history(user_id)
            new_exchange = {
                'query_text': question,
                'answer_text': answer,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Add new exchange and keep last 5
            history.insert(0, new_exchange)
            history = history[:5]
            
            self.conversation_cache[user_id] = history
            
            logger.debug(f"Saved conversation for user {user_id}")
            
        except Exception as e:
            logger.warning(f"Failed to save conversation for user {user_id}: {e}")

# Global LLM client instance
llm_client = LLMClient() 