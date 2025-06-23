import os
import signal
import multiprocessing as mp
from multiprocessing import Process
import uvicorn
from dotenv import load_dotenv

# Use relative imports for proper module structure
from .backend.logger import get_logger, setup_multiprocessing_logging
from .bot.discord_bot import run_discord_bot

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

def run_backend():
    """Run the FastAPI backend server."""
    try:
        # Setup multiprocessing logging
        setup_multiprocessing_logging()
        
        # Import the FastAPI app with relative import
        from .backend.api import app
        
        # Get configuration
        host = os.getenv("BACKEND_HOST", "0.0.0.0")
        port = int(os.getenv("BACKEND_PORT", 8000))
        log_level = os.getenv("LOG_LEVEL", "info").lower()
        
        logger.info(f"Starting FastAPI backend on {host}:{port}")
        
        # Run the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=log_level,
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start backend: {e}")
        raise

def run_bot():
    """Run the Discord bot."""
    try:
        # Setup multiprocessing logging
        setup_multiprocessing_logging()
        
        logger.info("Starting Discord bot")
        
        # Run the Discord bot
        run_discord_bot()
        
    except Exception as e:
        logger.error(f"Failed to start Discord bot: {e}")
        raise

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    
def validate_environment():
    """Validate all required environment variables are set."""
    required_vars = [
        "DISCORD_TOKEN",
        "OPENAI_API_KEY", 
        "PINECONE_API_KEY",
        "BACKEND_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please check your .env file and ensure all required variables are set.")
        logger.error("Copy env.example to .env and fill in your API keys:")
        for var in missing_vars:
            logger.error(f"  {var}=your_{var.lower()}_here")
        return False
    
    return True

def main():
    """Main entry point that starts both backend and bot processes."""
    try:
        # Validate environment variables first
        if not validate_environment():
            return
        
        logger.info("Starting VITA Discord AI Knowledge Assistant")
        logger.info("Press Ctrl+C to stop the application")
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start both processes
        backend_process = Process(target=run_backend, name="VITA-Backend")
        bot_process = Process(target=run_bot, name="VITA-Bot")
        
        # Start processes
        backend_process.start()
        bot_process.start()
        
        logger.info("Both processes started successfully")
        logger.info(f"Backend PID: {backend_process.pid}")
        logger.info(f"Bot PID: {bot_process.pid}")
        logger.info("Backend available at: http://localhost:8000")
        logger.info("Backend health check: http://localhost:8000/health")
        
        try:
            # Wait for processes to complete
            backend_process.join()
            bot_process.join()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        
        # Ensure both processes are terminated
        if backend_process.is_alive():
            logger.info("Terminating backend process...")
            backend_process.terminate()
            backend_process.join(timeout=5)
            if backend_process.is_alive():
                logger.warning("Force killing backend process...")
                backend_process.kill()
        
        if bot_process.is_alive():
            logger.info("Terminating bot process...")
            bot_process.terminate()
            bot_process.join(timeout=5)
            if bot_process.is_alive():
                logger.warning("Force killing bot process...")
                bot_process.kill()
        
        logger.info("VITA Discord AI Knowledge Assistant stopped")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

if __name__ == "__main__":
    # Use fork method on Unix systems to preserve environment changes
    import sys
    if sys.platform != 'win32':
        mp.set_start_method('fork', force=True)
    main() 