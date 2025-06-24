from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
from collections import deque
import structlog
from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

@dataclass
class PerformanceMetrics:
    """Performance metrics for individual operations."""
    operation: str
    duration_ms: float
    success: bool
    message_id: Optional[str] = None
    file_size_bytes: Optional[int] = None
    chunks_generated: Optional[int] = None
    vectors_stored: Optional[int] = None
    error_type: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class VitaMetricsCollector:
    """Comprehensive metrics collection for VITA system."""
    
    def __init__(self):
        # Create custom registry to avoid conflicts
        self.registry = CollectorRegistry()
        
        # Core ingestion metrics
        self.messages_processed = Counter(
            'vita_messages_processed_total',
            'Total number of messages processed',
            ['status', 'channel_type'],
            registry=self.registry
        )
        
        self.processing_duration = Histogram(
            'vita_processing_duration_seconds',
            'Time spent processing messages',
            ['operation', 'status'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self.registry
        )
        
        self.file_processing_duration = Histogram(
            'vita_file_processing_duration_seconds',
            'Time spent processing file attachments',
            ['file_type', 'status'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.embeddings_generated = Counter(
            'vita_embeddings_generated_total',
            'Total number of embeddings generated',
            ['source_type'],
            registry=self.registry
        )
        
        self.vectors_stored = Counter(
            'vita_vectors_stored_total',
            'Total number of vectors stored in Pinecone',
            ['status'],
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'vita_errors_total',
            'Total errors by type',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Performance tracking
        self.recent_metrics = deque(maxlen=1000)  # Keep last 1000 metrics
        
    def record_message_processing(self, duration: float, success: bool, 
                                 channel_type: str = "text"):
        """Record message processing metrics."""
        status = "success" if success else "failure"
        self.messages_processed.labels(status=status, channel_type=channel_type).inc()
        self.processing_duration.labels(operation="message_ingestion", status=status).observe(duration)
        
        # Store for aggregation
        metric = PerformanceMetrics(
            operation="message_processing",
            duration_ms=duration * 1000,
            success=success
        )
        self.recent_metrics.append(metric)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from recent metrics."""
        if not self.recent_metrics:
            return {}
        
        # Calculate aggregates
        total_ops = len(self.recent_metrics)
        successful_ops = sum(1 for m in self.recent_metrics if m.success)
        success_rate = (successful_ops / total_ops) * 100
        
        durations = [m.duration_ms for m in self.recent_metrics]
        avg_duration = sum(durations) / len(durations)
        
        return {
            "total_operations": total_ops,
            "success_rate": success_rate,
            "average_duration_ms": avg_duration,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def export_metrics(self) -> str:
        """Export Prometheus metrics."""
        return generate_latest(self.registry).decode('utf-8')

# Global metrics instance
vita_metrics = VitaMetricsCollector() 