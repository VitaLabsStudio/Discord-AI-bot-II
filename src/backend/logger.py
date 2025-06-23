import logging
import os
from typing import Dict
from dotenv import load_dotenv
import multiprocessing_logging

# Load environment variables
load_dotenv()

# Global logger cache to prevent duplicates
_loggers: Dict[str, logging.Logger] = {}

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with centralized configuration.
    
    Args:
        name: The logger name
        
    Returns:
        Configured logger instance
    """
    # Return cached logger if it exists
    if name in _loggers:
        return _loggers[name]
    
    # Create new logger
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers
    if not logger.handlers:
        # Get log level from environment
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s'
        )
        
        # Create and configure handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Prevent propagation to avoid duplicate messages
        logger.propagate = False
    
    # Cache the logger
    _loggers[name] = logger
    
    return logger

def setup_multiprocessing_logging():
    """Setup logging for multiprocessing environments."""
    multiprocessing_logging.install_mp_handler() 