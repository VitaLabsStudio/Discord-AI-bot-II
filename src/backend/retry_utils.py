import time
import asyncio
from functools import wraps
from typing import Callable, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APITimeoutError, APIConnectionError
from .logger import get_logger

logger = get_logger(__name__)

# Define retryable exceptions for OpenAI
OPENAI_RETRYABLE_EXCEPTIONS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    ConnectionError,
    TimeoutError
)

def retry_on_api_error(
    max_attempts: int = 5,
    min_wait: float = 10.0,  # Increased from 1.0 to 10.0 for more conservative approach
    max_wait: float = 180.0,  # Increased from 60.0 to 180.0 for longer waits
    exponential_base: int = 2
):
    """
    Decorator for retrying API calls with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds
        exponential_base: Base for exponential backoff
        
    Returns:
        Decorated function with retry logic
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=min_wait,
            min=min_wait,
            max=max_wait,
            exp_base=exponential_base
        ),
        retry=retry_if_exception_type(OPENAI_RETRYABLE_EXCEPTIONS),
        before_sleep=lambda retry_state: logger.warning(
            f"API call failed (attempt {retry_state.attempt_number}/{max_attempts}), "
            f"retrying in {retry_state.next_action.sleep} seconds: {retry_state.outcome.exception()}"
        ),
        reraise=True
    )

def retry_on_ratelimit(func: Callable) -> Callable:
    """
    Simple decorator for retrying functions on rate limits.
    This is a simplified version for backward compatibility.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with retry logic
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except OPENAI_RETRYABLE_EXCEPTIONS as e:
                if attempt == max_retries:
                    logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                    raise
                
                wait_time = min(2 ** attempt, 30)  # Exponential backoff, max 30 seconds
                logger.warning(
                    f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {wait_time} seconds: {e}"
                )
                await asyncio.sleep(wait_time)
            except Exception as e:
                # For non-retryable exceptions, fail immediately
                logger.error(f"Function {func.__name__} failed with non-retryable error: {e}")
                raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except OPENAI_RETRYABLE_EXCEPTIONS as e:
                if attempt == max_retries:
                    logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                    raise
                
                wait_time = min(2 ** attempt, 30)  # Exponential backoff, max 30 seconds
                logger.warning(
                    f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {wait_time} seconds: {e}"
                )
                time.sleep(wait_time)
            except Exception as e:
                # For non-retryable exceptions, fail immediately
                logger.error(f"Function {func.__name__} failed with non-retryable error: {e}")
                raise
    
    # Return appropriate wrapper based on whether function is async
    import inspect
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

class CircuitBreaker:
    """
    Simple circuit breaker pattern for API calls.
    Prevents cascading failures by temporarily disabling failing services.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker moving to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - service temporarily unavailable")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                logger.info("Circuit breaker reset to CLOSED state")
            self.failure_count = 0
            
            return result
            
        except Exception:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")
            
            raise

# Global circuit breakers for different services
openai_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=300)  # 5 minutes
pinecone_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=180)  # 3 minutes 