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

# Global database instance
vita_db = VitaDatabase()

# Compatibility functions for existing code
def is_processed(message_id: str) -> bool:
    """Check if a message has been processed (compatibility function)."""
    return vita_db.is_message_processed(message_id)

def mark_processed(message_id: str, channel_id: str = None, user_id: str = None):
    """Mark a message as processed (compatibility function)."""
    vita_db.mark_message_processed(message_id, channel_id, user_id) 