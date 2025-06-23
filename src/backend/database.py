import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Text, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError
from .logger import get_logger

logger = get_logger(__name__)

# SQLAlchemy Base
Base = declarative_base()

class ProcessedMessage(Base):
    """Table for tracking processed Discord messages."""
    __tablename__ = "processed_messages"
    
    message_id = Column(String, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    channel_id = Column(String, index=True)
    user_id = Column(String, index=True)
    
    def __repr__(self):
        return f"<ProcessedMessage(message_id='{self.message_id}', timestamp='{self.timestamp}')>"

class UserFeedback(Base):
    """Table for storing user feedback on AI responses."""
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=False)
    is_helpful = Column(Boolean, nullable=False)
    user_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    confidence_score = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<UserFeedback(id={self.id}, is_helpful={self.is_helpful}, user_id='{self.user_id}')>"

class GraphNode(Base):
    """Table for storing knowledge graph entities."""
    __tablename__ = "graph_nodes"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String, nullable=False, index=True)  # e.g., "Project", "Person"
    name = Column(String, nullable=False, index=True)   # e.g., "Project Phoenix", "John Doe"
    properties = Column(Text, nullable=True)            # JSON blob for extra info (renamed from metadata)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<GraphNode(id={self.id}, label='{self.label}', name='{self.name}')>"

class GraphEdge(Base):
    """Table for storing knowledge graph relationships."""
    __tablename__ = "graph_edges"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(Integer, nullable=False, index=True)      # Foreign key to graph_nodes
    target_id = Column(Integer, nullable=False, index=True)      # Foreign key to graph_nodes
    relationship = Column(String, nullable=False, index=True)    # e.g., "manages", "depends_on"
    properties = Column(Text, nullable=True)                     # JSON blob for extra info (renamed from metadata)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    message_id = Column(String, nullable=True, index=True)       # Track which message created this relationship
    
    def __repr__(self):
        return f"<GraphEdge(id={self.id}, source_id={self.source_id}, target_id={self.target_id}, relationship='{self.relationship}')>"

class ConversationHistory(Base):
    """Table for storing conversation context between users and VITA."""
    __tablename__ = "conversation_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<ConversationHistory(id={self.id}, user_id='{self.user_id}')>"

class ThematicDigest(Base):
    """Table for storing generated thematic summaries."""
    __tablename__ = "thematic_digests"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    time_period_start = Column(DateTime, nullable=False)
    time_period_end = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    cluster_info = Column(Text, nullable=True)  # JSON blob with clustering details
    
    def __repr__(self):
        return f"<ThematicDigest(id={self.id}, title='{self.title}')>"

class VitaDatabase:
    """Database manager for VITA Discord AI bot."""
    
    def __init__(self, db_path: str = "vita_data.db"):
        self.db_path = db_path
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,
            connect_args={"check_same_thread": False}
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        self._create_tables()
        logger.info(f"Database initialized at {db_path}")
    
    def _create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def is_message_processed(self, message_id: str) -> bool:
        """
        Check if a message has been processed.
        
        Args:
            message_id: Discord message ID
            
        Returns:
            True if message has been processed
        """
        with self.get_session() as session:
            try:
                result = session.query(ProcessedMessage).filter(
                    ProcessedMessage.message_id == message_id
                ).first()
                return result is not None
            except Exception as e:
                logger.error(f"Failed to check if message {message_id} is processed: {e}")
                return False
    
    def mark_message_processed(self, message_id: str, channel_id: str = None, user_id: str = None):
        """
        Mark a message as processed.
        
        Args:
            message_id: Discord message ID
            channel_id: Optional channel ID
            user_id: Optional user ID
        """
        with self.get_session() as session:
            try:
                # Check if already exists
                if self.is_message_processed(message_id):
                    logger.debug(f"Message {message_id} already marked as processed")
                    return
                
                processed_msg = ProcessedMessage(
                    message_id=message_id,
                    channel_id=channel_id,
                    user_id=user_id
                )
                session.add(processed_msg)
                session.commit()
                logger.debug(f"Marked message {message_id} as processed")
                
            except IntegrityError:
                # Message already exists, this is fine
                session.rollback()
                logger.debug(f"Message {message_id} already processed (integrity error)")
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to mark message {message_id} as processed: {e}")
                raise
    
    def get_processed_count(self) -> int:
        """
        Get the total number of processed messages.
        
        Returns:
            Number of processed messages
        """
        with self.get_session() as session:
            try:
                count = session.query(ProcessedMessage).count()
                return count
            except Exception as e:
                logger.error(f"Failed to get processed message count: {e}")
                return 0
    
    def record_user_feedback(self, query_text: str, answer_text: str, 
                           is_helpful: bool, user_id: str, 
                           confidence_score: float = None) -> int:
        """
        Record user feedback on an AI response.
        
        Args:
            query_text: Original user query
            answer_text: AI-generated answer
            is_helpful: Whether the user found the answer helpful
            user_id: Discord user ID
            confidence_score: Optional confidence score of the answer
            
        Returns:
            Feedback record ID
        """
        with self.get_session() as session:
            try:
                feedback = UserFeedback(
                    query_text=query_text,
                    answer_text=answer_text,
                    is_helpful=is_helpful,
                    user_id=user_id,
                    confidence_score=confidence_score
                )
                session.add(feedback)
                session.commit()
                session.refresh(feedback)
                
                logger.info(f"Recorded feedback from user {user_id}: {'helpful' if is_helpful else 'not helpful'}")
                return feedback.id
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to record user feedback: {e}")
                raise
    
    def get_feedback_stats(self) -> dict:
        """
        Get feedback statistics.
        
        Returns:
            Dictionary with feedback statistics
        """
        with self.get_session() as session:
            try:
                total_feedback = session.query(UserFeedback).count()
                helpful_feedback = session.query(UserFeedback).filter(
                    UserFeedback.is_helpful == True
                ).count()
                
                helpful_percentage = (helpful_feedback / total_feedback * 100) if total_feedback > 0 else 0
                
                return {
                    "total_feedback": total_feedback,
                    "helpful_feedback": helpful_feedback,
                    "unhelpful_feedback": total_feedback - helpful_feedback,
                    "helpful_percentage": round(helpful_percentage, 2)
                }
                
            except Exception as e:
                logger.error(f"Failed to get feedback stats: {e}")
                return {
                    "total_feedback": 0,
                    "helpful_feedback": 0,
                    "unhelpful_feedback": 0,
                    "helpful_percentage": 0.0
                }
    
    def cleanup_old_processed_messages(self, days: int = 30) -> int:
        """
        Clean up old processed message records.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of records deleted
        """
        with self.get_session() as session:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                deleted_count = session.query(ProcessedMessage).filter(
                    ProcessedMessage.timestamp < cutoff_date
                ).delete()
                
                session.commit()
                logger.info(f"Cleaned up {deleted_count} old processed message records")
                return deleted_count
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to cleanup old processed messages: {e}")
                return 0
    
    def migrate_from_json(self, json_file_path: str) -> int:
        """
        Migrate processed messages from old JSON format.
        
        Args:
            json_file_path: Path to the JSON file
            
        Returns:
            Number of messages migrated
        """
        if not os.path.exists(json_file_path):
            logger.info(f"No JSON file found at {json_file_path}, skipping migration")
            return 0
        
        try:
            import json
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                message_ids = data.get('processed_messages', [])
            
            migrated_count = 0
            with self.get_session() as session:
                for message_id in message_ids:
                    try:
                        if not self.is_message_processed(message_id):
                            processed_msg = ProcessedMessage(message_id=message_id)
                            session.add(processed_msg)
                            migrated_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to migrate message {message_id}: {e}")
                        continue
                
                session.commit()
                logger.info(f"Migrated {migrated_count} messages from JSON to SQLite")
                return migrated_count
                
        except Exception as e:
            logger.error(f"Failed to migrate from JSON: {e}")
            return 0
    
    # Knowledge Graph Methods
    
    def create_or_get_node(self, label: str, name: str, metadata: dict = None) -> int:
        """
        Create a new graph node or get existing one.
        
        Args:
            label: Node type (e.g., "Project", "Person")
            name: Node name (e.g., "Project Phoenix", "John Doe")
            metadata: Optional metadata dictionary
            
        Returns:
            Node ID
        """
        with self.get_session() as session:
            try:
                # Check if node already exists
                existing_node = session.query(GraphNode).filter(
                    GraphNode.label == label,
                    GraphNode.name == name
                ).first()
                
                if existing_node:
                    # Update properties if provided
                    if metadata:
                        import json
                        existing_node.properties = json.dumps(metadata)
                        existing_node.updated_at = datetime.utcnow()
                        session.commit()
                    return existing_node.id
                
                # Create new node
                import json
                node = GraphNode(
                    label=label,
                    name=name,
                    properties=json.dumps(metadata) if metadata else None
                )
                session.add(node)
                session.commit()
                session.refresh(node)
                
                logger.debug(f"Created graph node: {label}:{name}")
                return node.id
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to create/get graph node {label}:{name}: {e}")
                raise
    
    def create_edge(self, source_id: int, target_id: int, relationship: str, 
                   metadata: dict = None, message_id: str = None) -> int:
        """
        Create a relationship edge between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship: Relationship type (e.g., "manages", "depends_on")
            metadata: Optional metadata dictionary
            message_id: Optional message ID that created this relationship
            
        Returns:
            Edge ID
        """
        with self.get_session() as session:
            try:
                # Check if edge already exists
                existing_edge = session.query(GraphEdge).filter(
                    GraphEdge.source_id == source_id,
                    GraphEdge.target_id == target_id,
                    GraphEdge.relationship == relationship
                ).first()
                
                if existing_edge:
                    logger.debug(f"Edge already exists: {source_id} -{relationship}-> {target_id}")
                    return existing_edge.id
                
                # Create new edge
                import json
                edge = GraphEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relationship=relationship,
                    properties=json.dumps(metadata) if metadata else None,
                    message_id=message_id
                )
                session.add(edge)
                session.commit()
                session.refresh(edge)
                
                logger.debug(f"Created graph edge: {source_id} -{relationship}-> {target_id}")
                return edge.id
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to create graph edge: {e}")
                raise
    
    def query_graph(self, query_type: str, **kwargs) -> list:
        """
        Query the knowledge graph.
        
        Args:
            query_type: Type of query ("nodes", "edges", "relationships")
            **kwargs: Query parameters
            
        Returns:
            List of query results
        """
        with self.get_session() as session:
            try:
                if query_type == "nodes":
                    query = session.query(GraphNode)
                    if "label" in kwargs:
                        query = query.filter(GraphNode.label == kwargs["label"])
                    if "name" in kwargs:
                        query = query.filter(GraphNode.name.ilike(f"%{kwargs['name']}%"))
                    return query.all()
                
                elif query_type == "edges":
                    query = session.query(GraphEdge)
                    if "relationship" in kwargs:
                        query = query.filter(GraphEdge.relationship == kwargs["relationship"])
                    if "source_id" in kwargs:
                        query = query.filter(GraphEdge.source_id == kwargs["source_id"])
                    if "target_id" in kwargs:
                        query = query.filter(GraphEdge.target_id == kwargs["target_id"])
                    return query.all()
                
                elif query_type == "relationships":
                    # Get nodes with their relationships
                    node_id = kwargs.get("node_id")
                    if not node_id:
                        return []
                    
                    # Get outgoing relationships
                    outgoing = session.query(GraphEdge, GraphNode).join(
                        GraphNode, GraphEdge.target_id == GraphNode.id
                    ).filter(GraphEdge.source_id == node_id).all()
                    
                    # Get incoming relationships
                    incoming = session.query(GraphEdge, GraphNode).join(
                        GraphNode, GraphEdge.source_id == GraphNode.id
                    ).filter(GraphEdge.target_id == node_id).all()
                    
                    return {"outgoing": outgoing, "incoming": incoming}
                
                return []
                
            except Exception as e:
                logger.error(f"Failed to query graph: {e}")
                return []
    
    # Conversation History Methods
    
    def save_conversation(self, user_id: str, query_text: str, answer_text: str):
        """
        Save a conversation exchange.
        
        Args:
            user_id: Discord user ID
            query_text: User's question
            answer_text: VITA's response
        """
        with self.get_session() as session:
            try:
                conversation = ConversationHistory(
                    user_id=user_id,
                    query_text=query_text,
                    answer_text=answer_text
                )
                session.add(conversation)
                session.commit()
                
                # Keep only last 10 conversations per user
                old_conversations = session.query(ConversationHistory).filter(
                    ConversationHistory.user_id == user_id
                ).order_by(ConversationHistory.timestamp.desc()).offset(10).all()
                
                for old_conv in old_conversations:
                    session.delete(old_conv)
                
                session.commit()
                logger.debug(f"Saved conversation for user {user_id}")
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save conversation: {e}")
    
    def get_conversation_history(self, user_id: str, limit: int = 5) -> list:
        """
        Get recent conversation history for a user.
        
        Args:
            user_id: Discord user ID
            limit: Number of recent conversations to retrieve
            
        Returns:
            List of recent conversations
        """
        with self.get_session() as session:
            try:
                conversations = session.query(ConversationHistory).filter(
                    ConversationHistory.user_id == user_id
                ).order_by(ConversationHistory.timestamp.desc()).limit(limit).all()
                
                return conversations
                
            except Exception as e:
                logger.error(f"Failed to get conversation history: {e}")
                return []
    
    # Thematic Digest Methods
    
    def save_thematic_digest(self, title: str, summary: str, 
                           time_period_start: datetime, time_period_end: datetime,
                           cluster_info: dict = None) -> int:
        """
        Save a thematic digest.
        
        Args:
            title: Digest title
            summary: Digest summary text
            time_period_start: Start of analysis period
            time_period_end: End of analysis period
            cluster_info: Optional clustering information
            
        Returns:
            Digest ID
        """
        with self.get_session() as session:
            try:
                import json
                digest = ThematicDigest(
                    title=title,
                    summary=summary,
                    time_period_start=time_period_start,
                    time_period_end=time_period_end,
                    cluster_info=json.dumps(cluster_info) if cluster_info else None
                )
                session.add(digest)
                session.commit()
                session.refresh(digest)
                
                logger.info(f"Saved thematic digest: {title}")
                return digest.id
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save thematic digest: {e}")
                raise
    
    def get_recent_digests(self, limit: int = 10) -> list:
        """
        Get recent thematic digests.
        
        Args:
            limit: Number of digests to retrieve
            
        Returns:
            List of recent digests
        """
        with self.get_session() as session:
            try:
                digests = session.query(ThematicDigest).order_by(
                    ThematicDigest.created_at.desc()
                ).limit(limit).all()
                
                return digests
                
            except Exception as e:
                logger.error(f"Failed to get recent digests: {e}")
                return []

# Global database instance
vita_db = VitaDatabase()

# Compatibility functions for existing code
def is_processed(message_id: str) -> bool:
    """Check if a message has been processed (compatibility function)."""
    return vita_db.is_message_processed(message_id)

def mark_processed(message_id: str, channel_id: str = None, user_id: str = None):
    """Mark a message as processed (compatibility function)."""
    vita_db.mark_message_processed(message_id, channel_id, user_id) 