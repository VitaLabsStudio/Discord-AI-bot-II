import os
import signal
import multiprocessing as mp
from multiprocessing import Process
import uvicorn
import socket
import psutil
from dotenv import load_dotenv

# Use relative imports for proper module structure
from .backend.logger import get_logger, setup_multiprocessing_logging
from .bot.discord_bot import run_discord_bot

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

def check_port_available(port: int) -> bool:
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

def kill_existing_processes():
    """Kill any existing VITA processes to prevent conflicts."""
    current_pid = os.getpid()
    killed_count = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['pid'] == current_pid:
                continue
                
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any('src.main' in str(cmd) for cmd in cmdline):
                logger.info(f"Killing existing VITA process {proc.info['pid']}")
                proc.kill()
                killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if killed_count > 0:
        logger.info(f"Killed {killed_count} existing VITA processes")
        import time
        time.sleep(2)  # Give time for processes to die

def run_backend():
    """Run the FastAPI backend server."""
    try:
        # Setup multiprocessing logging
        setup_multiprocessing_logging()
        
        # Check if port is available
        port = int(os.getenv("BACKEND_PORT", 8000))
        if not check_port_available(port):
            logger.error(f"Port {port} is already in use. Cannot start backend.")
            return
        
        # Import the FastAPI app with relative import
        from .backend.api import app
        
        # Get configuration
        host = os.getenv("BACKEND_HOST", "0.0.0.0")
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
        
        # Clean up any existing processes
        kill_existing_processes()
        
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