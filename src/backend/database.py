import os
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Text, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError
import uuid
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
    # v6.2: Add content column for searchable text
    content = Column(Text, nullable=True, index=True)
    # v6.1: Add content hash for duplicate detection
    content_hash = Column(String, unique=True, index=True, nullable=True)
    
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

class EvidenceChain(Base):
    """Table for storing evidence chains for traceable reasoning."""
    __tablename__ = "evidence_chains"
    
    chain_id = Column(String, primary_key=True, index=True)  # UUID
    user_query = Column(Text, nullable=False)
    reasoning_plan = Column(Text, nullable=True)  # JSON of LLM-generated plan
    evidence_data = Column(Text, nullable=True)   # JSON blob with source message IDs, etc.
    final_narrative = Column(Text, nullable=True)
    was_successful = Column(Boolean, default=False, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    user_id = Column(String, nullable=True, index=True)
    
    def __repr__(self):
        return f"<EvidenceChain(chain_id='{self.chain_id}', successful={self.was_successful})>"

class GraphNode(Base):
    """Table for storing knowledge graph entities with lifecycle management."""
    __tablename__ = "graph_nodes"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String, nullable=False, index=True)  # e.g., "Project", "Person"
    name = Column(String, nullable=False, index=True)   # e.g., "Project Phoenix", "John Doe"
    properties = Column(Text, nullable=True)            # JSON blob for extra info
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # v5.1 Lifecycle Management Fields
    status = Column(String, default="active", nullable=False, index=True)  # active, superseded, archived
    version = Column(Integer, default=1, nullable=False)
    last_accessed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<GraphNode(id={self.id}, label='{self.label}', name='{self.name}', status='{self.status}')>"

class GraphEdge(Base):
    """Table for storing knowledge graph relationships."""
    __tablename__ = "graph_edges"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(Integer, nullable=False, index=True)      # Foreign key to graph_nodes
    target_id = Column(Integer, nullable=False, index=True)      # Foreign key to graph_nodes
    relationship = Column(String, nullable=False, index=True)    # e.g., "manages", "depends_on", "supersedes"
    properties = Column(Text, nullable=True)                     # JSON blob for extra info
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    message_id = Column(String, nullable=True, index=True)       # Track which message created this relationship
    
    def __repr__(self):
        return f"<GraphEdge(id={self.id}, source_id={self.source_id}, target_id={self.target_id}, relationship='{self.relationship}')>"

class PlaybookUsage(Base):
    """Table for tracking playbook and SOP usage with feedback correlation."""
    __tablename__ = "playbook_usage"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    playbook_node_id = Column(Integer, nullable=False, index=True)  # Reference to graph_nodes
    user_query = Column(Text, nullable=False)
    user_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    feedback_id = Column(Integer, nullable=True, index=True)  # Reference to user_feedback
    was_helpful = Column(Boolean, nullable=True)  # Derived from feedback
    
    def __repr__(self):
        return f"<PlaybookUsage(id={self.id}, playbook_node_id={self.playbook_node_id}, helpful={self.was_helpful})>"

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

# v6.1: New table for attachment versioning and traceability
class Attachment(Base):
    """Table for tracking file attachments with versioning."""
    __tablename__ = "attachments"
    
    attachment_id = Column(String, primary_key=True, index=True)  # SHA-256 hash of file content
    original_filename = Column(String, nullable=False)
    file_size_bytes = Column(Integer, nullable=False) 
    mime_type = Column(String, nullable=True)
    first_seen_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    download_url = Column(String, nullable=True)  # For reference
    
    def __repr__(self):
        return f"<Attachment(attachment_id='{self.attachment_id}', filename='{self.original_filename}')>"

# v7.1: New tables for adaptive relevance and viewpoint processing
class ChannelMetrics(Base):
    """Table for tracking channel-specific relevance metrics."""
    __tablename__ = "channel_metrics"
    
    channel_id = Column(String, primary_key=True, index=True)
    total_messages_processed = Column(Integer, default=0, nullable=False)
    irrelevant_messages_count = Column(Integer, default=0, nullable=False)
    dynamic_relevance_threshold = Column(Float, default=0.3, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<ChannelMetrics(channel_id='{self.channel_id}', threshold={self.dynamic_relevance_threshold})>"

class ContentViewpoint(Base):
    """Table for storing multi-viewpoint analysis of critical content."""
    __tablename__ = "content_viewpoints"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    original_message_id = Column(String, nullable=False, index=True)
    viewpoint_type = Column(String, nullable=False, index=True)  # executive, technical, actionable, qa_synthetic
    content = Column(Text, nullable=False)
    retrieval_weight = Column(Float, default=1.0, nullable=False)
    target_roles = Column(Text, nullable=True)  # JSON array of target roles
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<ContentViewpoint(id={self.id}, viewpoint='{self.viewpoint_type}', message='{self.original_message_id}')>"

class RelevanceFeedback(Base):
    """Table for tracking relevance classification feedback for learning."""
    __tablename__ = "relevance_feedback"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    predicted_category = Column(String, nullable=False)  # CRITICAL, HIGH, MEDIUM, LOW, IRRELEVANT
    actual_usefulness = Column(Integer, nullable=True)  # 1=helpful, 0=not helpful, null=no feedback
    feedback_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<RelevanceFeedback(id={self.id}, predicted='{self.predicted_category}', actual={self.actual_usefulness})>"

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
                    UserFeedback.is_helpful
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
    
    # Evidence Chain Methods (v5.1)
    
    def create_evidence_chain(self, user_query: str, user_id: str = None) -> str:
        """
        Create a new evidence chain for query traceability.
        
        Args:
            user_query: The user's original query
            user_id: Optional user ID
            
        Returns:
            Chain ID (UUID)
        """
        with self.get_session() as session:
            try:
                chain_id = str(uuid.uuid4())
                
                evidence_chain = EvidenceChain(
                    chain_id=chain_id,
                    user_query=user_query,
                    user_id=user_id
                )
                session.add(evidence_chain)
                session.commit()
                
                logger.debug(f"Created evidence chain {chain_id} for query: {user_query[:50]}...")
                return chain_id
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to create evidence chain: {e}")
                raise
    
    def update_evidence_chain(self, chain_id: str, reasoning_plan: str = None, 
                             evidence_data: dict = None, final_narrative: str = None, 
                             was_successful: bool = None) -> bool:
        """
        Update an evidence chain with reasoning steps and results.
        
        Args:
            chain_id: The chain ID to update
            reasoning_plan: JSON string of the reasoning plan
            evidence_data: Dictionary of evidence (will be JSON encoded)
            final_narrative: The final answer provided
            was_successful: Whether the chain was completed successfully
            
        Returns:
            True if update was successful
        """
        with self.get_session() as session:
            try:
                chain = session.query(EvidenceChain).filter(
                    EvidenceChain.chain_id == chain_id
                ).first()
                
                if not chain:
                    logger.warning(f"Evidence chain {chain_id} not found")
                    return False
                
                if reasoning_plan is not None:
                    chain.reasoning_plan = reasoning_plan
                if evidence_data is not None:
                    import json
                    chain.evidence_data = json.dumps(evidence_data)
                if final_narrative is not None:
                    chain.final_narrative = final_narrative
                if was_successful is not None:
                    chain.was_successful = was_successful
                
                session.commit()
                logger.debug(f"Updated evidence chain {chain_id}")
                return True
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to update evidence chain {chain_id}: {e}")
                return False
    
    def get_failed_evidence_chains(self, days: int = 7) -> list:
        """
        Get evidence chains that failed to resolve (knowledge gaps).
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of failed evidence chains
        """
        with self.get_session() as session:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                failed_chains = session.query(EvidenceChain).filter(
                    not EvidenceChain.was_successful,
                    EvidenceChain.timestamp >= cutoff_date
                ).order_by(EvidenceChain.timestamp.desc()).all()

                
                return failed_chains
                
            except Exception as e:
                logger.error(f"Failed to get failed evidence chains: {e}")
                return []

    # Enhanced Knowledge Graph Methods (v5.1)
    
    def supersede_node(self, old_node_id: int, new_node_id: int, message_id: str = None) -> bool:
        """
        Mark a node as superseded and create a supersedes relationship.
        
        Args:
            old_node_id: ID of the node being superseded
            new_node_id: ID of the new node that supersedes it
            message_id: Optional message ID that triggered the supersession
            
        Returns:
            True if successful
        """
        with self.get_session() as session:
            try:
                # Update old node status
                old_node = session.query(GraphNode).filter(GraphNode.id == old_node_id).first()
                if old_node:
                    old_node.status = "superseded"
                    old_node.updated_at = datetime.utcnow()
                
                # Create supersedes relationship
                import json
                edge = GraphEdge(
                    source_id=new_node_id,
                    target_id=old_node_id,
                    relationship="supersedes",
                    message_id=message_id,
                    properties=json.dumps({"superseded_at": datetime.utcnow().isoformat()})
                )
                session.add(edge)
                session.commit()
                
                logger.info(f"Node {old_node_id} superseded by node {new_node_id}")
                return True
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to supersede node {old_node_id}: {e}")
                return False
    
    def get_active_nodes(self, label: str = None, name: str = None) -> list:
        """
        Get active (non-superseded) nodes with optional filtering.
        
        Args:
            label: Optional label filter
            name: Optional name filter
            
        Returns:
            List of active nodes
        """
        with self.get_session() as session:
            try:
                query = session.query(GraphNode).filter(GraphNode.status == "active")
                
                if label:
                    query = query.filter(GraphNode.label == label)
                if name:
                    query = query.filter(GraphNode.name.ilike(f"%{name}%"))
                
                # Update last_accessed_at for returned nodes
                nodes = query.all()
                for node in nodes:
                    node.last_accessed_at = datetime.utcnow()
                
                session.commit()
                return nodes
                
            except Exception as e:
                logger.error(f"Failed to get active nodes: {e}")
                return []
    
    def get_superseded_nodes(self, days: int = 30) -> list:
        """
        Get nodes that were superseded in the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of superseded nodes with their superseding nodes
        """
        with self.get_session() as session:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Get superseded nodes and their superseding relationships
                superseded_info = session.query(GraphNode, GraphEdge, GraphNode.alias()).join(
                    GraphEdge, GraphNode.id == GraphEdge.target_id
                ).join(
                    GraphNode.alias(), GraphEdge.source_id == GraphNode.alias().id
                ).filter(
                    GraphNode.status == "superseded",
                    GraphEdge.relationship == "supersedes",
                    GraphNode.updated_at >= cutoff_date
                ).all()
                
                return superseded_info
                
            except Exception as e:
                logger.error(f"Failed to get superseded nodes: {e}")
                return []

    # Playbook Usage Tracking Methods (v5.1)
    
    def record_playbook_usage(self, playbook_node_id: int, user_query: str, 
                             user_id: str, feedback_id: int = None) -> int:
        """
        Record usage of a playbook or SOP.
        
        Args:
            playbook_node_id: ID of the playbook node in the knowledge graph
            user_query: The user's query that used this playbook
            user_id: User ID
            feedback_id: Optional feedback ID to correlate with feedback
            
        Returns:
            Usage record ID
        """
        with self.get_session() as session:
            try:
                usage = PlaybookUsage(
                    playbook_node_id=playbook_node_id,
                    user_query=user_query,
                    user_id=user_id,
                    feedback_id=feedback_id
                )
                session.add(usage)
                session.commit()
                session.refresh(usage)
                
                logger.debug(f"Recorded playbook usage for node {playbook_node_id}")
                return usage.id
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to record playbook usage: {e}")
                raise
    
    def get_playbook_feedback_summary(self, days: int = 30) -> list:
        """
        Get playbook usage feedback summary for review.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            List of playbooks with their feedback statistics
        """
        with self.get_session() as session:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Get playbook usage with feedback correlation
                playbook_stats = session.query(
                    PlaybookUsage.playbook_node_id,
                    GraphNode.name,
                    GraphNode.label,
                    session.query(UserFeedback.is_helpful).filter(
                        UserFeedback.id == PlaybookUsage.feedback_id
                    ).label('feedback_helpful')
                ).join(
                    GraphNode, PlaybookUsage.playbook_node_id == GraphNode.id
                ).outerjoin(
                    UserFeedback, PlaybookUsage.feedback_id == UserFeedback.id
                ).filter(
                    PlaybookUsage.timestamp >= cutoff_date
                ).all()
                
                # Aggregate feedback by playbook
                playbook_summary = {}
                for stat in playbook_stats:
                    node_id = stat.playbook_node_id
                    if node_id not in playbook_summary:
                        playbook_summary[node_id] = {
                            'node_id': node_id,
                            'name': stat.name,
                            'label': stat.label,
                            'total_usage': 0,
                            'positive_feedback': 0,
                            'negative_feedback': 0,
                            'no_feedback': 0
                        }
                    
                    playbook_summary[node_id]['total_usage'] += 1
                    
                    if stat.feedback_helpful is True:
                        playbook_summary[node_id]['positive_feedback'] += 1
                    elif stat.feedback_helpful is False:
                        playbook_summary[node_id]['negative_feedback'] += 1
                    else:
                        playbook_summary[node_id]['no_feedback'] += 1
                
                return list(playbook_summary.values())
                
            except Exception as e:
                logger.error(f"Failed to get playbook feedback summary: {e}")
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

    # v6.1: Duplicate Guard Methods
    
    def check_content_duplicate(self, content_hash: str) -> bool:
        """
        Check if content with this hash has already been processed.
        
        Args:
            content_hash: SHA-256 hash of the content
            
        Returns:
            True if duplicate exists
        """
        with self.get_session() as session:
            try:
                result = session.query(ProcessedMessage).filter(
                    ProcessedMessage.content_hash == content_hash
                ).first()
                return result is not None
            except Exception as e:
                logger.error(f"Failed to check content duplicate for hash {content_hash}: {e}")
                return False
    
    def mark_message_processed_with_hash(self, message_id: str, content_hash: str, 
                                       channel_id: str = None, user_id: str = None, content: str = None):
        """
        Mark a message as processed with content hash for duplicate detection.
        
        Args:
            message_id: Discord message ID
            content_hash: SHA-256 hash of the content
            channel_id: Optional channel ID
            user_id: Optional user ID
            content: Optional message content for searchability
        """
        with self.get_session() as session:
            try:
                # Check if already exists
                if self.is_message_processed(message_id):
                    logger.debug(f"Message {message_id} already marked as processed")
                    return
                
                processed_msg = ProcessedMessage(
                    message_id=message_id,
                    content_hash=content_hash,
                    channel_id=channel_id,
                    user_id=user_id,
                    content=content[:10000] if content else None  # Limit content to 10k chars
                )
                session.add(processed_msg)
                session.commit()
                logger.debug(f"Marked message {message_id} as processed with hash {content_hash[:8]}...")
                
            except IntegrityError as e:
                # Content hash already exists - this is a duplicate
                session.rollback()
                if "content_hash" in str(e):
                    logger.info(f"Duplicate content detected for message {message_id} (hash: {content_hash[:8]}...)")
                else:
                    logger.debug(f"Message {message_id} already processed (integrity error)")
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to mark message {message_id} as processed: {e}")
                raise
    
    # v6.1: Attachment Management Methods
    
    def register_attachment(self, attachment_id: str, original_filename: str, 
                          file_size_bytes: int, mime_type: str = None, 
                          download_url: str = None) -> bool:
        """
        Register a new attachment or confirm existing one.
        
        Args:
            attachment_id: SHA-256 hash of file content
            original_filename: Original filename
            file_size_bytes: File size in bytes
            mime_type: MIME type of the file
            download_url: Original download URL
            
        Returns:
            True if new attachment, False if already exists
        """
        with self.get_session() as session:
            try:
                # Check if attachment already exists
                existing = session.query(Attachment).filter(
                    Attachment.attachment_id == attachment_id
                ).first()
                
                if existing:
                    logger.debug(f"Attachment {attachment_id[:8]}... already registered")
                    return False
                
                # Register new attachment
                attachment = Attachment(
                    attachment_id=attachment_id,
                    original_filename=original_filename,
                    file_size_bytes=file_size_bytes,
                    mime_type=mime_type,
                    download_url=download_url
                )
                session.add(attachment)
                session.commit()
                
                logger.info(f"Registered new attachment: {original_filename} ({attachment_id[:8]}...)")
                return True
                
            except IntegrityError:
                # Attachment already exists
                session.rollback()
                logger.debug(f"Attachment {attachment_id[:8]}... already exists (integrity error)")
                return False
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to register attachment {attachment_id}: {e}")
                raise
    
    def get_attachment_info(self, attachment_id: str) -> Optional[dict]:
        """
        Get attachment information by ID.
        
        Args:
            attachment_id: Attachment ID (SHA-256 hash)
            
        Returns:
            Dictionary with attachment info or None if not found
        """
        with self.get_session() as session:
            try:
                attachment = session.query(Attachment).filter(
                    Attachment.attachment_id == attachment_id
                ).first()
                
                if attachment:
                    return {
                        "attachment_id": attachment.attachment_id,
                        "original_filename": attachment.original_filename,
                        "file_size_bytes": attachment.file_size_bytes,
                        "mime_type": attachment.mime_type,
                        "first_seen_at": attachment.first_seen_at.isoformat(),
                        "download_url": attachment.download_url
                    }
                return None
                
            except Exception as e:
                logger.error(f"Failed to get attachment info for {attachment_id}: {e}")
                return None

    # v7.1: Channel Metrics Methods
    
    def get_channel_threshold(self, channel_id: str) -> float:
        """Get dynamic relevance threshold for a channel."""
        with self.get_session() as session:
            try:
                metrics = session.query(ChannelMetrics).filter(
                    ChannelMetrics.channel_id == channel_id
                ).first()
                
                if metrics:
                    return metrics.dynamic_relevance_threshold
                else:
                    # Create default metrics for new channel
                    default_threshold = 0.3
                    new_metrics = ChannelMetrics(
                        channel_id=channel_id,
                        dynamic_relevance_threshold=default_threshold
                    )
                    session.add(new_metrics)
                    session.commit()
                    return default_threshold
                    
            except Exception as e:
                logger.error(f"Failed to get channel threshold for {channel_id}: {e}")
                return 0.3  # Default fallback
    
    def update_channel_metrics(self, channel_id: str, was_relevant: bool) -> bool:
        """Update channel metrics with new message processing result."""
        with self.get_session() as session:
            try:
                metrics = session.query(ChannelMetrics).filter(
                    ChannelMetrics.channel_id == channel_id
                ).first()
                
                if not metrics:
                    metrics = ChannelMetrics(channel_id=channel_id)
                    session.add(metrics)
                
                metrics.total_messages_processed += 1
                if not was_relevant:
                    metrics.irrelevant_messages_count += 1
                
                # Auto-adjust threshold if we have enough data
                if metrics.total_messages_processed >= 100:
                    irrelevant_rate = metrics.irrelevant_messages_count / metrics.total_messages_processed
                    
                    # Adjust threshold based on irrelevant rate
                    if irrelevant_rate > 0.7:  # Too much noise, raise threshold
                        metrics.dynamic_relevance_threshold = min(0.8, metrics.dynamic_relevance_threshold + 0.05)
                    elif irrelevant_rate < 0.3:  # Too restrictive, lower threshold
                        metrics.dynamic_relevance_threshold = max(0.1, metrics.dynamic_relevance_threshold - 0.05)
                
                session.commit()
                return True
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to update channel metrics for {channel_id}: {e}")
                return False
    
    def store_content_viewpoint(self, message_id: str, viewpoint_type: str, 
                               content: str, retrieval_weight: float = 1.0, 
                               target_roles: list = None) -> int:
        """Store a viewpoint analysis of content."""
        with self.get_session() as session:
            try:
                import json
                viewpoint = ContentViewpoint(
                    original_message_id=message_id,
                    viewpoint_type=viewpoint_type,
                    content=content,
                    retrieval_weight=retrieval_weight,
                    target_roles=json.dumps(target_roles) if target_roles else None
                )
                session.add(viewpoint)
                session.commit()
                session.refresh(viewpoint)
                
                logger.debug(f"Stored {viewpoint_type} viewpoint for message {message_id}")
                return viewpoint.id
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to store content viewpoint: {e}")
                raise
    
    def get_viewpoints_for_message(self, message_id: str) -> list:
        """Get all viewpoints for a specific message."""
        with self.get_session() as session:
            try:
                viewpoints = session.query(ContentViewpoint).filter(
                    ContentViewpoint.original_message_id == message_id
                ).all()
                
                result = []
                for vp in viewpoints:
                    import json
                    result.append({
                        "id": vp.id,
                        "viewpoint_type": vp.viewpoint_type,
                        "content": vp.content,
                        "retrieval_weight": vp.retrieval_weight,
                        "target_roles": json.loads(vp.target_roles) if vp.target_roles else [],
                        "created_at": vp.created_at.isoformat()
                    })
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to get viewpoints for message {message_id}: {e}")
                return []
    
    def record_relevance_feedback(self, message_id: str, user_id: str, 
                                 predicted_category: str, actual_usefulness: int = None) -> int:
        """Record feedback on relevance classification accuracy."""
        with self.get_session() as session:
            try:
                feedback = RelevanceFeedback(
                    message_id=message_id,
                    user_id=user_id,
                    predicted_category=predicted_category,
                    actual_usefulness=actual_usefulness
                )
                session.add(feedback)
                session.commit()
                session.refresh(feedback)
                
                logger.debug(f"Recorded relevance feedback for message {message_id}")
                return feedback.id
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to record relevance feedback: {e}")
                raise
    
    def get_relevance_feedback_stats(self, days: int = 30) -> dict:
        """Get relevance classification accuracy statistics."""
        with self.get_session() as session:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Get feedback with actual usefulness ratings
                feedback_with_ratings = session.query(RelevanceFeedback).filter(
                    RelevanceFeedback.feedback_timestamp >= cutoff_date,
                    RelevanceFeedback.actual_usefulness.isnot(None)
                ).all()
                
                if not feedback_with_ratings:
                    return {"total_feedback": 0, "accuracy_by_category": {}}
                
                # Calculate accuracy by predicted category
                accuracy_stats = {}
                for category in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'IRRELEVANT']:
                    category_feedback = [f for f in feedback_with_ratings if f.predicted_category == category]
                    if category_feedback:
                        correct_predictions = sum(1 for f in category_feedback if f.actual_usefulness == 1)
                        accuracy = correct_predictions / len(category_feedback)
                        accuracy_stats[category] = {
                            "total": len(category_feedback),
                            "correct": correct_predictions,
                            "accuracy": round(accuracy, 3)
                        }
                
                return {
                    "total_feedback": len(feedback_with_ratings),
                    "accuracy_by_category": accuracy_stats
                }
                
            except Exception as e:
                logger.error(f"Failed to get relevance feedback stats: {e}")
                return {"total_feedback": 0, "accuracy_by_category": {}}

# Global database instance
vita_db = VitaDatabase()

# Compatibility functions for existing code
def is_processed(message_id: str) -> bool:
    """Check if a message has been processed (compatibility function)."""
    return vita_db.is_message_processed(message_id)

def mark_processed(message_id: str, channel_id: str = None, user_id: str = None):
    """Mark a message as processed (compatibility function)."""
    vita_db.mark_message_processed(message_id, channel_id, user_id) 