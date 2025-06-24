import os
from typing import List
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Header, Request
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
from .schemas import IngestRequest, QueryRequest, QueryResponse, BatchIngestRequest, ProgressResponse
from .ingestion import run_ingestion_task, run_batch_ingestion_task
from .database import vita_db, is_processed
from .query_router import query_router
from .embedding import embedding_manager
from .permissions import permission_manager
from .llm_client import llm_client
from .progress_tracker import progress_tracker
from .logger import get_logger
from datetime import datetime
import aiohttp
import json
from fastapi.responses import JSONResponse

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# v6.1: Custom Prometheus metrics
vita_llm_tokens_total = Counter(
    'vita_llm_tokens_total',
    'Total tokens used by LLM calls',
    ['model_name', 'operation']
)

vita_estimated_cost_usd_total = Counter(
    'vita_estimated_cost_usd_total',
    'Total estimated cost in USD for LLM calls',
    ['model_name']
)

vita_knowledge_graph_nodes_total = Gauge(
    'vita_knowledge_graph_nodes_total',
    'Total number of nodes in knowledge graph'
)

vita_vector_count_total = Gauge(
    'vita_vector_count_total',
    'Total number of vectors in Pinecone'
)

vita_query_confidence = Histogram(
    'vita_query_confidence',
    'Distribution of query confidence scores',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# v6.1: Token Budget Guard Middleware
class TokenBudgetMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce monthly token budget limits."""
    
    def __init__(self, app):
        super().__init__(app)
        self.monthly_budget = float(os.getenv("MONTHLY_BUDGET_USD", "100.0"))
        self.llm_endpoints = ["/query", "/debug/query"]
    
    async def dispatch(self, request: Request, call_next):
        # Check if this is an LLM endpoint
        if request.url.path in self.llm_endpoints:
            # Get current cost from Counter - use ._value.values() for prometheus_client
            try:
                current_cost = sum(vita_estimated_cost_usd_total._value.values())
            except (AttributeError, TypeError):
                # Fallback if Counter structure is different
                current_cost = 0.0
            
            if current_cost >= self.monthly_budget:
                logger.warning(f"Monthly budget exceeded: ${current_cost:.2f} >= ${self.monthly_budget}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"Monthly budget exceeded (${current_cost:.2f}/${self.monthly_budget}). Contact administrator."
                    }
                )
        
        response = await call_next(request)
        return response

# Initialize FastAPI app
app = FastAPI(
    title="VITA Discord AI Knowledge Assistant v6.1",
    description="Production-grade backend API for Discord AI knowledge management",
    version="6.1.0"
)

# v6.1: Add Prometheus instrumentation
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# v6.1: Add budget middleware
app.add_middleware(TokenBudgetMiddleware)

# Security
security = HTTPBearer()

def track_llm_usage(model: str, tokens: int, operation: str):
    """Track LLM token usage and estimated costs."""
    vita_llm_tokens_total.labels(model_name=model, operation=operation).inc(tokens)
    
    # Estimate cost based on model pricing (approximate)
    cost_per_1k_tokens = {
        "gpt-4o": 0.03,
        "gpt-4o-mini": 0.0015,
        "text-embedding-3-small": 0.0001
    }
    
    cost_estimate = (tokens / 1000) * cost_per_1k_tokens.get(model, 0.01)
    vita_estimated_cost_usd_total.labels(model_name=model).inc(cost_estimate)
    
    logger.debug(f"LLM usage tracked: {model} used {tokens} tokens (${cost_estimate:.4f}) for {operation}")

def update_metrics():
    """Update gauge metrics with current system state."""
    try:
        # Update knowledge graph node count
        with vita_db.get_session() as session:
            node_count = session.query(vita_db.GraphNode).filter(vita_db.GraphNode.status == 'active').count()
            vita_knowledge_graph_nodes_total.set(node_count)
        
        # Update vector count (approximate from Pinecone stats)
        try:
            stats = embedding_manager.index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            vita_vector_count_total.set(vector_count)
        except Exception as e:
            logger.warning(f"Failed to update vector count metric: {e}")
            
    except Exception as e:
        logger.error(f"Failed to update metrics: {e}")

async def verify_api_key(x_api_key: str = Header(...)):
    """
    Verify API key from request headers.
    
    Args:
        x_api_key: API key from X-API-Key header
        
    Returns:
        True if valid
        
    Raises:
        HTTPException: If API key is invalid
    """
    expected_key = os.getenv("BACKEND_API_KEY")
    if not expected_key:
        raise HTTPException(status_code=500, detail="Backend API key not configured")
    
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "VITA Discord AI Knowledge Assistant API", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "openai": "configured" if os.getenv("OPENAI_API_KEY") else "not_configured",
            "pinecone": "configured" if os.getenv("PINECONE_API_KEY") else "not_configured"
        }
    }

@app.post("/ingest", status_code=202)
async def ingest_message(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_api_key)
):
    """
    Ingest a Discord message for knowledge storage.
    
    CRITICAL FIX: Uses BackgroundTasks for async processing.
    Returns 202 Accepted immediately after queuing the task.
    
    Args:
        request: IngestRequest with message data
        background_tasks: FastAPI background tasks
        
    Returns:
        202 Accepted response
    """
    try:
        # Log the incoming request for verification
        logger.info(f"Received ingestion request for message {request.message_id}")
        
        # Check for idempotency
        if is_processed(request.message_id):
            logger.info(f"Message {request.message_id} already processed")
            return {
                "status": "already_processed",
                "message_id": request.message_id
            }
        
        # Add ingestion task to background queue
        background_tasks.add_task(run_ingestion_task, request)
        
        logger.info(f"Queued ingestion task for message {request.message_id}")
        
        return {
            "status": "accepted",
            "message_id": request.message_id,
            "message": "Ingestion task queued for processing"
        }
        
    except Exception as e:
        logger.error(f"Failed to queue ingestion task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue ingestion task: {str(e)}")

@app.post("/batch_ingest", status_code=202)
async def batch_ingest_messages(
    request: BatchIngestRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_api_key)
):
    """
    Batch ingest multiple Discord messages with progress tracking.
    
    Args:
        request: BatchIngestRequest with list of messages
        background_tasks: FastAPI background tasks
        
    Returns:
        202 Accepted response with batch_id for tracking
    """
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        if len(request.messages) > 1000:
            raise HTTPException(status_code=400, detail="Batch size too large (max 1000 messages)")
        
        # Convert to IngestRequest objects if needed with validation
        ingest_requests = []
        validation_errors = []
        
        for i, msg in enumerate(request.messages):
            try:
                if isinstance(msg, dict):
                    # Validate required fields before conversion
                    required_fields = ["message_id", "channel_id", "user_id", "content", "timestamp"]
                    missing_fields = [field for field in required_fields if field not in msg]
                    if missing_fields:
                        validation_errors.append(f"Message {i}: Missing fields {missing_fields}")
                        continue
                    
                    # Convert dict to IngestRequest
                    ingest_req = IngestRequest(**msg)
                else:
                    # Already an IngestRequest object
                    ingest_req = msg
                ingest_requests.append(ingest_req)
            except Exception as e:
                validation_errors.append(f"Message {i}: {str(e)}")
        
        if validation_errors:
            error_summary = f"Validation errors in batch: {'; '.join(validation_errors[:5])}"
            if len(validation_errors) > 5:
                error_summary += f" (and {len(validation_errors) - 5} more errors)"
            raise HTTPException(status_code=400, detail=error_summary)
        
        if not ingest_requests:
            raise HTTPException(status_code=400, detail="No valid messages to process")
        
        # Create batch tracker
        batch_id = progress_tracker.create_batch(len(ingest_requests))
        
        # Add batch ingestion task to background queue
        background_tasks.add_task(run_batch_ingestion_task, ingest_requests, batch_id)
        
        logger.info(f"Queued batch ingestion task for {len(ingest_requests)} messages with batch_id: {batch_id}")
        
        return {
            "status": "accepted",
            "batch_id": batch_id,
            "message_count": len(ingest_requests),
            "valid_messages": len(ingest_requests),
            "total_submitted": len(request.messages),
            "message": "Batch ingestion task queued for processing",
            "progress_endpoint": f"/progress/{batch_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to queue batch ingestion task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue batch ingestion task: {str(e)}")

@app.get("/progress/{batch_id}", response_model=ProgressResponse)
async def get_progress(
    batch_id: str,
    _: bool = Depends(verify_api_key)
):
    """
    Get real-time progress for a batch ingestion operation.
    
    Args:
        batch_id: Unique batch identifier
        
    Returns:
        Progress information and recent logs
    """
    try:
        progress = progress_tracker.get_progress(batch_id)
        
        if not progress:
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
        
        return ProgressResponse(
            progress=progress,
            message=f"Batch progress: {progress.processed_count}/{progress.total_messages} processed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get progress for batch {batch_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get progress: {str(e)}")

@app.post("/debug/query", response_model=QueryResponse)
async def debug_query_knowledge(
    request: QueryRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Debug endpoint that shows exactly what happens during query processing.
    """
    debug_info = {
        "step": 1,
        "received_query": request.question,
        "user_id": request.user_id,
        "roles": request.roles,
        "channel_id": request.channel_id
    }
    logger.info(f"DEBUG: Step 1 - Received query: {debug_info}")
    
    # Test query router
    routed_response = query_router.route_query(request.question, request.user_id)
    debug_info["step"] = 2
    debug_info["router_response"] = routed_response
    logger.info(f"DEBUG: Step 2 - Router response: {debug_info}")
    
    if routed_response:
        # This is the problem path - query is being routed to FAQ
        logger.info(f"DEBUG: Query was routed to FAQ with source: {routed_response.get('source')}")
        return QueryResponse(
            answer=f"[DEBUG] Query routed to FAQ: {routed_response['answer']}",
            citations=routed_response["citations"],
            confidence=routed_response["confidence"]
        )
    else:
        # This is the correct path - should go to AI pipeline
        logger.info("DEBUG: Query will go to AI pipeline")
        return QueryResponse(
            answer="[DEBUG] Query correctly routed to AI pipeline (this would return real results)",
            citations=[],
            confidence=1.0
        )

@app.post("/query", response_model=QueryResponse)
async def query_knowledge(
    request: QueryRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Query the knowledge base using enhanced RAG pipeline with evidence chain tracking.
    
    Args:
        request: QueryRequest with user question and context
        
    Returns:
        QueryResponse with answer and citations
    """
    try:
        logger.info(f"Processing query from user {request.user_id}")
        
        # First, try to route the query to FAQ cache
        routed_response = query_router.route_query(request.question, request.user_id)
        if routed_response:
            # Return cached response immediately
            response = QueryResponse(
                answer=routed_response["answer"],
                citations=routed_response["citations"],
                confidence=routed_response["confidence"]
            )
            logger.info(f"Query answered by router ({routed_response['source']})")
            return response
        
        # If not routed, proceed with AI pipeline
        logger.info(f"Query routed to AI pipeline for user {request.user_id}")
        
        # Try enhanced v5.1 pipeline first, fall back to basic RAG if rate limited
        try:
            return await _execute_enhanced_query(request)
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                logger.warning(f"Rate limit detected, falling back to basic RAG for user {request.user_id}")
                return await _execute_basic_query(request)
            else:
                raise e
        
    except Exception as e:
        logger.error(f"Failed to process query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

async def _execute_enhanced_query(request: QueryRequest) -> QueryResponse:
    """Execute the full v5.1 enhanced query with evidence chains."""
    # v5.1: Create evidence chain for traceability
    chain_id = vita_db.create_evidence_chain(request.question, request.user_id)
    evidence_data = {
        "retrieved_docs": [],
        "kg_nodes": [],
        "reasoning_steps": [],
        "fallback_reasons": []
    }
    
    try:
        # Generate multi-hop reasoning plan
        reasoning_plan = await _generate_reasoning_plan(request.question, request.roles)
        vita_db.update_evidence_chain(chain_id, reasoning_plan=reasoning_plan)
        
        # Execute iterative query pipeline
        answer_result = await _execute_multi_hop_query(
            request, reasoning_plan, evidence_data, chain_id
        )
        
        # Update evidence chain with final results
        vita_db.update_evidence_chain(
            chain_id,
            evidence_data=evidence_data,
            final_narrative=answer_result["answer"],
            was_successful=answer_result["was_successful"]
        )
        
        response = QueryResponse(
            answer=answer_result["answer"],
            citations=answer_result["citations"],
            confidence=answer_result["confidence"]
        )
        
        logger.info(f"Generated enhanced response with evidence chain {chain_id}")
        return response
        
    except Exception as e:
        # Fallback mechanism: proceed with partial evidence
        logger.warning(f"Evidence chain {chain_id} encountered error: {e}")
        evidence_data["fallback_reasons"].append(str(e))
        
        # Generate fallback response
        fallback_result = await _generate_fallback_response(request, evidence_data)
        
        vita_db.update_evidence_chain(
            chain_id,
            evidence_data=evidence_data,
            final_narrative=fallback_result["answer"],
            was_successful=False
        )
        
        return QueryResponse(
            answer=fallback_result["answer"],
            citations=fallback_result["citations"],
            confidence=fallback_result["confidence"]
        )

async def _execute_basic_query(request: QueryRequest) -> QueryResponse:
    """Execute basic RAG query without v5.1 features to avoid rate limits."""
    try:
        logger.info(f"Executing basic RAG query for user {request.user_id}")
        
        # Create permission filter
        permission_filter = permission_manager.create_permission_filter(
            request.roles, 
            request.channel_id
        )
        
        # Do direct vector search
        similar_docs = await embedding_manager.query_similar(
            request.question,
            top_k=request.top_k,
            filter_dict=permission_filter
        )
        
        # Generate citations with duplicate removal
        citations = []
        seen_message_ids = set()
        for doc in similar_docs:
            metadata = doc.get("metadata", {})
            message_id = metadata.get("message_id", "")
            
            # Skip duplicates by message_id
            if message_id and message_id in seen_message_ids:
                continue
            seen_message_ids.add(message_id)
            
            citation = {
                "message_id": message_id,
                "channel_id": metadata.get("channel_id", ""),
                "user_id": metadata.get("user_id", ""),
                "timestamp": metadata.get("timestamp", ""),
                "content_preview": metadata.get("content", "")[:200] + "..." if len(metadata.get("content", "")) > 200 else metadata.get("content", ""),
                "score": doc.get("score", 0.0)
            }
            citations.append(citation)
        
        if not citations:
            return QueryResponse(
                answer="I couldn't find any relevant information in your server's content for this question. Try rephrasing your query or asking about a different topic.",
                citations=[],
                confidence=0.0
            )
        
        # Generate answer using basic LLM call with retry logic
        try:
            # Convert docs for LLM
            context_docs = []
            for doc in similar_docs:
                context_docs.append({
                    "metadata": doc.get("metadata", {}),
                    "score": doc.get("score", 0.0)
                })
            
            llm_result = await llm_client.generate_answer(
                question=request.question,
                context_documents=context_docs,
                user_id=request.user_id,
                user_roles=request.roles
            )
            
            return QueryResponse(
                answer=llm_result["answer"],
                citations=citations,
                confidence=llm_result["confidence"]
            )
            
        except Exception as llm_error:
            # If LLM also fails, return a basic response with citations
            logger.warning(f"LLM call failed in basic mode: {llm_error}")
            
            # Create a simple response based on the most relevant citation
            if citations:
                top_citation = citations[0]
                answer = f"Based on the most relevant information I found:\n\n{top_citation['content_preview']}\n\nThis information comes from a message in your server. You can view more details in the sources below."
                confidence = min(top_citation['score'], 0.6)  # Cap confidence for basic response
            else:
                answer = "I found some relevant content but couldn't generate a detailed response. Please check the sources below."
                confidence = 0.3
            
            return QueryResponse(
                answer=answer,
                citations=citations,
                confidence=confidence
            )
        
    except Exception as e:
        logger.error(f"Basic query execution failed: {e}")
        return QueryResponse(
            answer="I apologize, but I encountered difficulties processing your question. Please try again in a moment or rephrase your query.",
            citations=[],
            confidence=0.0
        )

async def _generate_reasoning_plan(question: str, roles: list = None) -> str:
    """
    Generate a multi-step reasoning plan for complex queries.
    
    v6.1: Uses cost-effective model for planning to save on costs.
    
    Args:
        question: User's question
        roles: User's roles for context
        
    Returns:
        JSON string of reasoning plan
    """
    try:
        prompt = f"""Analyze this question and create a step-by-step reasoning plan.

Question: {question}

Instructions:
1. Determine if this is a simple factual query or requires multi-hop reasoning
2. If multi-hop, break down into logical steps
3. Identify what types of evidence would be needed for each step
4. Consider dependencies between steps
5. Determine if any step requires precise graph query (use "graph_query" tool)

v6.1: If a step needs to query relationships or find connected entities, mark it as requiring "graph_query" tool.

Format your response as JSON:
{{
    "complexity": "simple|multi_hop",
    "steps": [
        {{
            "step_number": 1,
            "description": "What to find/verify",
            "evidence_type": "documents|knowledge_graph|both",
            "requires_tool": "graph_query|none",
            "tool_query": {{"match": {{"source": {{"label": "Person", "name": "John Doe"}}, "edge": {{"relationship": "manages"}}, "target": {{"label": "Project"}}}}, "return": ["target.name"]}},
            "depends_on": []
        }}
    ],
    "expected_evidence_types": ["message_content", "decisions", "relationships"]
}}
"""
        
        # v6.1: Use cost-effective model for planning
        response = await llm_client.client.chat.completions.create(
            model="gpt-4o-mini",  # Cost-effective model for planning
            messages=[
                {"role": "system", "content": "You are an expert query planner who creates structured reasoning plans. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        
        plan_text = response.choices[0].message.content.strip()
        logger.info(f"Generated reasoning plan using cost-effective model: {plan_text[:100]}...")
        
        return plan_text
        
    except Exception as e:
        logger.error(f"Failed to generate reasoning plan: {e}")
        # Return simple plan as fallback
        import json
        return json.dumps({
            "complexity": "simple",
            "steps": [{"step_number": 1, "description": "Direct search", "evidence_type": "documents", "requires_tool": "none", "depends_on": []}],
            "expected_evidence_types": ["message_content"]
        })

async def _execute_multi_hop_query(request: QueryRequest, reasoning_plan: str, 
                                  evidence_data: dict, chain_id: str) -> dict:
    """
    Execute multi-hop reasoning query with evidence tracking.
    
    Args:
        request: Original query request
        reasoning_plan: JSON reasoning plan
        evidence_data: Evidence tracking dictionary
        chain_id: Evidence chain ID
        
    Returns:
        Query result with answer, citations, and success status
    """
    try:
        import json
        plan = json.loads(reasoning_plan)
        
        # Create permission filter
        permission_filter = permission_manager.create_permission_filter(
            request.roles, 
            request.channel_id
        )
        
        all_citations = []
        reasoning_context = []
        
        # Execute each reasoning step
        for step in plan.get("steps", []):
            step_evidence = await _execute_reasoning_step(
                step, request.question, permission_filter, evidence_data
            )
            
            reasoning_context.append({
                "step": step["step_number"],
                "description": step["description"],
                "evidence_found": len(step_evidence["documents"]) > 0,
                "evidence_count": len(step_evidence["documents"])
            })
            
            all_citations.extend(step_evidence["citations"])
            evidence_data["reasoning_steps"].append(reasoning_context[-1])
        
        # Generate final answer with accumulated evidence
        if all_citations:
            # Combine all retrieved documents
            all_docs = []
            for citation in all_citations:
                all_docs.append({
                    "metadata": {
                        "message_id": citation["message_id"],
                        "content": citation["content_preview"],
                        "channel_id": citation["channel_id"],
                        "user_id": citation["user_id"],
                        "timestamp": citation["timestamp"]
                    },
                    "score": citation["score"]
                })
            
            llm_result = await llm_client.generate_answer(
                question=request.question,
                context_documents=all_docs,
                user_id=request.user_id,
                user_roles=request.roles
            )
            
            return {
                "answer": llm_result["answer"],
                "citations": all_citations,
                "confidence": llm_result["confidence"],
                "was_successful": True
            }
        else:
            # No evidence found - use fallback
            raise Exception("No evidence found for multi-hop reasoning")
            
    except Exception as e:
        logger.error(f"Multi-hop query execution failed: {e}")
        raise

async def _execute_reasoning_step(step: dict, original_question: str, 
                                 permission_filter: dict, evidence_data: dict) -> dict:
    """
    Execute a single reasoning step.
    
    v6.1: Enhanced with tool-calling capability for graph queries.
    
    Args:
        step: Step definition from reasoning plan
        original_question: Original user question
        permission_filter: Permission filter for documents
        evidence_data: Evidence tracking dictionary
        
    Returns:
        Step results with documents and citations
    """
    try:
        evidence_type = step.get("evidence_type", "documents")
        requires_tool = step.get("requires_tool", "none")
        step_query = f"{step['description']} (related to: {original_question})"
        
        documents = []
        citations = []
        
        # v6.1: Handle tool-calling for graph queries
        if requires_tool == "graph_query" and step.get("tool_query"):
            try:
                logger.info(f"Executing graph query tool for step {step.get('step_number', 'unknown')}")
                
                # Call the graph-query endpoint internally
                backend_url = "http://localhost:8000"  # Internal call
                api_key = os.getenv("BACKEND_API_KEY")
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{backend_url}/graph-query",
                        json=step["tool_query"],
                        headers={"X-API-Key": api_key}
                    ) as response:
                        if response.status == 200:
                            graph_results = await response.json()
                            
                            # Convert graph results to citations format
                            for result in graph_results.get("results", []):
                                citation = {
                                    "message_id": f"graph_query_{step.get('step_number', '')}",
                                    "channel_id": "knowledge_graph",
                                    "user_id": "graph_system",
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "content_preview": f"Graph data: {json.dumps(result)}",
                                    "score": 1.0,
                                    "reasoning_step": step["step_number"],
                                    "source_type": "knowledge_graph"
                                }
                                citations.append(citation)
                                evidence_data["kg_nodes"].append({
                                    "result": result,
                                    "step": step["step_number"],
                                    "tool_query": step["tool_query"]
                                })
                            
                            logger.info(f"Graph query returned {len(graph_results.get('results', []))} results")
                        else:
                            logger.warning(f"Graph query failed with status {response.status}")
                            
            except Exception as tool_error:
                logger.error(f"Graph query tool failed: {tool_error}")
                # Continue with document search as fallback
        
        if evidence_type in ["documents", "both"]:
            # Search vector database
            similar_docs = await embedding_manager.query_similar(
                step_query,
                top_k=5,
                filter_dict=permission_filter
            )
            
            # Track seen message IDs to prevent duplicates within this step
            seen_message_ids = set()
            for doc in similar_docs:
                metadata = doc.get("metadata", {})
                message_id = metadata.get("message_id", "")
                
                # Skip duplicates by message_id
                if message_id and message_id in seen_message_ids:
                    continue
                seen_message_ids.add(message_id)
                
                citation = {
                    "message_id": message_id,
                    "channel_id": metadata.get("channel_id", ""),
                    "user_id": metadata.get("user_id", ""),
                    "timestamp": metadata.get("timestamp", ""),
                    "content_preview": metadata.get("content", "")[:200] + "..." if len(metadata.get("content", "")) > 200 else metadata.get("content", ""),
                    "score": doc.get("score", 0.0),
                    "reasoning_step": step["step_number"],
                    "source_type": "vector_search"
                }
                citations.append(citation)
                documents.append(doc)
                evidence_data["retrieved_docs"].append(citation)
        
        if evidence_type in ["knowledge_graph", "both"] and requires_tool != "graph_query":
            # Legacy graph search for non-tool steps
            kg_nodes = vita_db.get_active_nodes()  # Could be enhanced with semantic search
            for node in kg_nodes[:3]:  # Limit to top 3 for now
                evidence_data["kg_nodes"].append({
                    "node_id": node.id,
                    "label": node.label,
                    "name": node.name,
                    "step": step["step_number"],
                    "source_type": "legacy_graph_search"
                })
        
        return {
            "documents": documents,
            "citations": citations,
            "step_number": step["step_number"]
        }
        
    except Exception as e:
        logger.error(f"Failed to execute reasoning step {step.get('step_number', 'unknown')}: {e}")
        return {"documents": [], "citations": [], "step_number": step.get("step_number", 0)}

async def _generate_fallback_response(request: QueryRequest, evidence_data: dict) -> dict:
    """
    Generate a fallback response when evidence chain fails.
    
    Args:
        request: Original query request
        evidence_data: Partial evidence collected
        
    Returns:
        Fallback response with explanation
    """
    try:
        # Create permission filter
        permission_filter = permission_manager.create_permission_filter(
            request.roles, 
            request.channel_id
        )
        
        # Do basic search as fallback
        similar_docs = await embedding_manager.query_similar(
            request.question,
            top_k=request.top_k,
            filter_dict=permission_filter
        )
        
        # Generate fallback citations with duplicate removal
        citations = []
        seen_message_ids = set()
        for doc in similar_docs:
            metadata = doc.get("metadata", {})
            message_id = metadata.get("message_id", "")
            
            # Skip duplicates by message_id
            if message_id and message_id in seen_message_ids:
                continue
            seen_message_ids.add(message_id)
            
            citation = {
                "message_id": message_id,
                "channel_id": metadata.get("channel_id", ""),
                "user_id": metadata.get("user_id", ""),
                "timestamp": metadata.get("timestamp", ""),
                "content_preview": metadata.get("content", "")[:200] + "..." if len(metadata.get("content", "")) > 200 else metadata.get("content", ""),
                "score": doc.get("score", 0.0)
            }
            citations.append(citation)
        
        # Generate answer with fallback prompt
        fallback_prompt = f"""Based on the available evidence, please answer this question: {request.question}

Note: The evidence for some parts of this query was incomplete. Please synthesize the answer based on the available evidence and explicitly state any uncertainties or missing links. If you cannot fully answer the question, suggest who might have the answer based on the available context.

Available Evidence:
{len(similar_docs)} relevant documents found
{len(evidence_data.get('reasoning_steps', []))} reasoning steps attempted
Fallback reasons: {', '.join(evidence_data.get('fallback_reasons', []))}
"""
        
        response = await llm_client.client.chat.completions.create(
            model=llm_client.chat_model,
            messages=[
                {"role": "system", "content": llm_client._create_system_prompt(request.roles)},
                {"role": "user", "content": fallback_prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        
        answer = response.choices[0].message.content.strip()
        
        return {
            "answer": answer,
            "citations": citations,
            "confidence": 0.3  # Lower confidence for fallback responses
        }
        
    except Exception as e:
        logger.error(f"Fallback response generation failed: {e}")
        return {
            "answer": "I apologize, but I encountered difficulties processing your question. Please try rephrasing your query or contact an administrator for assistance.",
            "citations": [],
            "confidence": 0.0
        }

@app.post("/feedback")
async def record_feedback(
    request: dict,
    _: bool = Depends(verify_api_key)
):
    """
    Record user feedback on AI responses.
    
    Args:
        request: Dictionary with feedback data
        
    Returns:
        Success confirmation
    """
    try:
        required_fields = ["query_text", "answer_text", "is_helpful", "user_id"]
        missing_fields = [field for field in required_fields if field not in request]
        
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {missing_fields}"
            )
        
        feedback_id = vita_db.record_user_feedback(
            query_text=request["query_text"],
            answer_text=request["answer_text"],
            is_helpful=request["is_helpful"],
            user_id=request["user_id"],
            confidence_score=request.get("confidence_score")
        )
        
        return {
            "message": "Feedback recorded successfully",
            "feedback_id": feedback_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")

@app.delete("/delete_message/{message_id}")
async def delete_message_vectors(
    message_id: str,
    _: bool = Depends(verify_api_key)
):
    """
    Delete all vectors associated with a message ID from Pinecone.
    
    Args:
        message_id: Discord message ID
        
    Returns:
        Success confirmation
    """
    try:
        # Query for vectors with this message_id
        query_response = embedding_manager.index.query(
            vector=[0.0] * 1536,  # Dummy vector
            top_k=10000,  # Large number to get all matches
            include_metadata=True,
            filter={"message_id": message_id}
        )
        
        if not query_response.matches:
            return {
                "message": f"No vectors found for message {message_id}",
                "deleted_count": 0
            }
        
        # Extract vector IDs to delete
        vector_ids = [match.id for match in query_response.matches]
        
        # Delete vectors from Pinecone
        embedding_manager.index.delete(ids=vector_ids)
        
        logger.info(f"Deleted {len(vector_ids)} vectors for message {message_id}")
        
        return {
            "message": f"Successfully deleted vectors for message {message_id}",
            "deleted_count": len(vector_ids)
        }
        
    except Exception as e:
        logger.error(f"Failed to delete vectors for message {message_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete vectors: {str(e)}")

@app.post("/digest")
async def generate_digest(
    request: dict,
    _: bool = Depends(verify_api_key)
):
    """
    Generate a thematic digest for the specified time period.
    
    Args:
        request: Dictionary with 'days' parameter (optional, default: 7)
        
    Returns:
        Thematic digest with key themes and summaries
    """
    try:
        from .analyzer import vita_analyzer
        
        days = request.get("days", 7)
        
        # Validate days parameter
        if not isinstance(days, int) or days < 1 or days > 30:
            raise HTTPException(
                status_code=400,
                detail="Days parameter must be an integer between 1 and 30"
            )
        
        digest = await vita_analyzer.generate_thematic_digest(days)
        
        return {
            "digest": digest,
            "message": f"Generated thematic digest for last {days} days"
        }
        
    except Exception as e:
        logger.error(f"Failed to generate digest: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate digest: {str(e)}")

@app.get("/graph/query")
async def query_knowledge_graph(
    query_type: str,
    label: str = None,
    name: str = None,
    relationship: str = None,
    node_id: int = None,
    _: bool = Depends(verify_api_key)
):
    """
    Query the knowledge graph.
    
    Args:
        query_type: Type of query ("nodes", "edges", "relationships")
        label: Optional node label filter
        name: Optional node name filter  
        relationship: Optional relationship type filter
        node_id: Optional node ID for relationship queries
        
    Returns:
        Query results from the knowledge graph
    """
    try:
        if query_type not in ["nodes", "edges", "relationships"]:
            raise HTTPException(
                status_code=400,
                detail="query_type must be 'nodes', 'edges', or 'relationships'"
            )
        
        # Build query parameters
        query_params = {}
        if label:
            query_params["label"] = label
        if name:
            query_params["name"] = name
        if relationship:
            query_params["relationship"] = relationship
        if node_id:
            query_params["node_id"] = node_id
        
        results = vita_db.query_graph(query_type, **query_params)
        
        # Convert results to dictionaries for JSON response
        if query_type == "relationships" and isinstance(results, dict):
            # Special handling for relationship queries
            response_data = {
                "outgoing": [{"edge": edge.__dict__, "node": node.__dict__} for edge, node in results.get("outgoing", [])],
                "incoming": [{"edge": edge.__dict__, "node": node.__dict__} for edge, node in results.get("incoming", [])]
            }
        else:
            # Standard list of objects
            response_data = [item.__dict__ for item in results]
        
        return {
            "results": response_data,
            "count": len(response_data) if isinstance(response_data, list) else sum(len(v) for v in response_data.values()),
            "query_type": query_type,
            "parameters": query_params
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to query knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to query knowledge graph: {str(e)}")

@app.get("/digests/recent")
async def get_recent_digests(
    limit: int = 10,
    _: bool = Depends(verify_api_key)
):
    """
    Get recent thematic digests.
    
    Args:
        limit: Number of digests to retrieve (max 50)
        
    Returns:
        List of recent digests
    """
    try:
        if limit < 1 or limit > 50:
            limit = min(max(limit, 1), 50)
        
        digests = vita_db.get_recent_digests(limit)
        
        # Convert to dictionaries for JSON response
        digests_data = []
        for digest in digests:
            digest_dict = digest.__dict__.copy()
            # Parse cluster_info JSON if present
            if digest_dict.get('cluster_info'):
                import json
                try:
                    digest_dict['cluster_info'] = json.loads(digest_dict['cluster_info'])
                except json.JSONDecodeError:
                    pass
            digests_data.append(digest_dict)
        
        return {
            "digests": digests_data,
            "count": len(digests_data)
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent digests: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent digests: {str(e)}")

@app.get("/stats")
async def get_stats(_: bool = Depends(verify_api_key)):
    """
    Get enhanced system statistics including feedback data and knowledge graph metrics.
    
    Returns:
        System statistics
    """
    try:
        feedback_stats = vita_db.get_feedback_stats()
        faq_stats = query_router.get_stats()
        
        # Get knowledge graph statistics
        total_nodes = len(vita_db.query_graph("nodes"))
        total_edges = len(vita_db.query_graph("edges"))
        
        # Get concept distribution
        concept_counts = {}
        from .ontology import ontology_manager
        for concept_name in ontology_manager.get_concept_names():
            nodes = vita_db.query_graph("nodes", label=concept_name)
            concept_counts[concept_name] = len(nodes)
        
        stats = {
            "processed_messages": vita_db.get_processed_count(),
            "feedback": feedback_stats,
            "faq_cache": faq_stats,
            "knowledge_graph": {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "concept_distribution": concept_counts
            },
            "system_status": "operational"
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# Admin endpoints for Dead Letter Queue management
@app.get("/admin/dlq/stats")
async def get_dlq_stats(_: bool = Depends(verify_api_key)):
    """
    Get dead letter queue statistics (Admin only).
    
    Returns:
        DLQ statistics including failure types and counts
    """
    try:
        from .dead_letter_queue import enhanced_dlq
        
        stats = enhanced_dlq.get_stats()
        
        return {
            "stats": stats,
            "message": "DLQ statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get DLQ stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get DLQ stats: {str(e)}")

@app.get("/admin/dlq/items")
async def get_dlq_items(
    failure_type: str = None,
    limit: int = 10,
    _: bool = Depends(verify_api_key)
):
    """
    Get dead letter queue items (Admin only).
    
    Args:
        failure_type: Optional filter by failure type
        limit: Maximum number of items to return
        
    Returns:
        List of DLQ items
    """
    try:
        from .dead_letter_queue import enhanced_dlq, FailureType
        
        # Validate failure type if provided
        failure_type_enum = None
        if failure_type:
            try:
                failure_type_enum = FailureType(failure_type.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid failure_type. Valid types: {[ft.value for ft in FailureType]}"
                )
        
        # Validate limit
        if limit < 1 or limit > 100:
            limit = min(max(limit, 1), 100)
        
        items = enhanced_dlq.get_items(failure_type=failure_type_enum, limit=limit)
        
        # Convert items to dictionaries for JSON response
        items_data = [item.to_dict() for item in items]
        
        return {
            "items": items_data,
            "count": len(items_data),
            "message": f"Retrieved {len(items_data)} DLQ items"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get DLQ items: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get DLQ items: {str(e)}")

@app.post("/admin/dlq/cleanup")
async def cleanup_dlq(
    request: dict,
    _: bool = Depends(verify_api_key)
):
    """
    Clean up old dead letter queue items (Admin only).
    
    Args:
        request: Dictionary with 'days' parameter
        
    Returns:
        Number of items removed
    """
    try:
        from .dead_letter_queue import enhanced_dlq
        
        days = request.get("days", 30)
        
        # Validate days parameter
        if not isinstance(days, int) or days < 1 or days > 365:
            raise HTTPException(
                status_code=400,
                detail="Days parameter must be an integer between 1 and 365"
            )
        
        removed_count = enhanced_dlq.cleanup_old_items(days)
        
        return {
            "removed_count": removed_count,
            "message": f"Cleaned up {removed_count} items older than {days} days"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cleanup DLQ: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup DLQ: {str(e)}")

@app.delete("/admin/dlq/items/{item_id}")
async def remove_dlq_item(
    item_id: str,
    _: bool = Depends(verify_api_key)
):
    """
    Remove a specific dead letter queue item (Admin only).
    
    Args:
        item_id: ID of the item to remove
        
    Returns:
        Success message
    """
    try:
        from .dead_letter_queue import enhanced_dlq
        
        success = enhanced_dlq.remove_item(item_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"DLQ item {item_id} not found")
        
        return {
            "message": f"Successfully removed DLQ item {item_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove DLQ item {item_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove DLQ item: {str(e)}")

@app.post("/admin/dlq/items/{item_id}/retry")
async def retry_dlq_item(
    item_id: str,
    _: bool = Depends(verify_api_key)
):
    """
    Mark a dead letter queue item as retried (Admin only).
    
    Args:
        item_id: ID of the item to mark as retried
        
    Returns:
        Success message
    """
    try:
        from .dead_letter_queue import enhanced_dlq
        
        # Get the item first
        item = enhanced_dlq.get_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail=f"DLQ item {item_id} not found")
        
        # Mark as retried
        success = enhanced_dlq.mark_retry(item_id)
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to mark item {item_id} as retried")
        
        return {
            "message": f"Successfully marked DLQ item {item_id} as retried",
            "retry_count": item.retry_count + 1
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry DLQ item {item_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retry DLQ item: {str(e)}")

# v5.1: Evidence Chain & Traceability Endpoints

@app.get("/evidence_chains/{chain_id}")
async def get_evidence_chain(
    chain_id: str,
    _: bool = Depends(verify_api_key)
):
    """
    Get detailed information about an evidence chain for query traceability.
    
    Args:
        chain_id: Evidence chain ID
        
    Returns:
        Evidence chain details with reasoning steps
    """
    try:
        with vita_db.get_session() as session:
            from .database import EvidenceChain
            chain = session.query(EvidenceChain).filter(
                EvidenceChain.chain_id == chain_id
            ).first()
            
            if not chain:
                raise HTTPException(status_code=404, detail=f"Evidence chain {chain_id} not found")
            
            # Parse JSON fields
            import json
            evidence_data = {}
            reasoning_plan = {}
            
            try:
                if chain.evidence_data:
                    evidence_data = json.loads(chain.evidence_data)
                if chain.reasoning_plan:
                    reasoning_plan = json.loads(chain.reasoning_plan)
            except json.JSONDecodeError:
                pass
            
            return {
                "chain_id": chain.chain_id,
                "user_query": chain.user_query,
                "user_id": chain.user_id,
                "reasoning_plan": reasoning_plan,
                "evidence_data": evidence_data,
                "final_narrative": chain.final_narrative,
                "was_successful": chain.was_successful,
                "timestamp": chain.timestamp.isoformat()
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get evidence chain {chain_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get evidence chain: {str(e)}")

@app.get("/evidence_chains/failed")
async def get_failed_evidence_chains(
    days: int = 7,
    _: bool = Depends(verify_api_key)
):
    """
    Get evidence chains that failed to resolve (knowledge gaps).
    
    Args:
        days: Number of days to look back (default: 7)
        
    Returns:
        List of failed evidence chains indicating knowledge gaps
    """
    try:
        if days < 1 or days > 30:
            days = min(max(days, 1), 30)
        
        failed_chains = vita_db.get_failed_evidence_chains(days)
        
        chains_data = []
        for chain in failed_chains:
            chains_data.append({
                "chain_id": chain.chain_id,
                "user_query": chain.user_query,
                "user_id": chain.user_id,
                "timestamp": chain.timestamp.isoformat(),
                "reasoning_plan": chain.reasoning_plan
            })
        
        return {
            "failed_chains": chains_data,
            "count": len(chains_data),
            "time_period_days": days,
            "message": f"Found {len(chains_data)} failed evidence chains in last {days} days"
        }
        
    except Exception as e:
        logger.error(f"Failed to get failed evidence chains: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get failed evidence chains: {str(e)}")

# v5.1: Knowledge Lifecycle Management Endpoints

@app.get("/knowledge/superseded")
async def get_superseded_knowledge(
    days: int = 30,
    _: bool = Depends(verify_api_key)
):
    """
    Get knowledge nodes that have been superseded recently.
    
    Args:
        days: Number of days to look back (default: 30)
        
    Returns:
        List of superseded knowledge with replacement information
    """
    try:
        if days < 1 or days > 90:
            days = min(max(days, 1), 90)
        
        superseded_nodes = vita_db.get_superseded_nodes(days)
        
        superseded_data = []
        for superseded_info in superseded_nodes:
            superseded_node, edge, superseding_node = superseded_info
            
            superseded_data.append({
                "superseded_node": {
                    "id": superseded_node.id,
                    "label": superseded_node.label,
                    "name": superseded_node.name,
                    "updated_at": superseded_node.updated_at.isoformat()
                },
                "superseding_node": {
                    "id": superseding_node.id,
                    "label": superseding_node.label,
                    "name": superseding_node.name,
                    "created_at": superseding_node.created_at.isoformat()
                },
                "relationship": edge.relationship,
                "message_id": edge.message_id
            })
        
        return {
            "superseded_knowledge": superseded_data,
            "count": len(superseded_data),
            "time_period_days": days
        }
        
    except Exception as e:
        logger.error(f"Failed to get superseded knowledge: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get superseded knowledge: {str(e)}")

@app.get("/knowledge/playbooks/review")
async def review_playbook_performance(
    _: bool = Depends(verify_api_key)
):
    """
    Review playbook and SOP performance based on user feedback.
    
    Returns:
        Playbook performance review with recommendations
    """
    try:
        from .analyzer import vita_analyzer
        
        review_results = await vita_analyzer.review_playbook_performance()
        
        return {
            "review_results": review_results,
            "message": f"Reviewed {review_results.get('total_playbooks_reviewed', 0)} playbooks, {review_results.get('flagged_playbooks', 0)} flagged"
        }
        
    except Exception as e:
        logger.error(f"Failed to review playbook performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to review playbook performance: {str(e)}")

@app.post("/knowledge/supersede")
async def supersede_knowledge(
    request: dict,
    _: bool = Depends(verify_api_key)
):
    """
    Manually supersede a knowledge node (Admin only).
    
    Args:
        request: Dictionary with old_node_id, new_node_id, message_id
        
    Returns:
        Supersession result
    """
    try:
        required_fields = ["old_node_id", "new_node_id"]
        missing_fields = [field for field in required_fields if field not in request]
        
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {missing_fields}"
            )
        
        success = vita_db.supersede_node(
            old_node_id=request["old_node_id"],
            new_node_id=request["new_node_id"],
            message_id=request.get("message_id")
        )
        
        if success:
            return {
                "message": f"Successfully superseded node {request['old_node_id']} with node {request['new_node_id']}",
                "old_node_id": request["old_node_id"],
                "new_node_id": request["new_node_id"]
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to supersede knowledge node"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to supersede knowledge: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to supersede knowledge: {str(e)}")

# v5.1: Predictive Intelligence & Strategic Advisory Endpoints

@app.post("/intelligence/downstream_risks")
async def detect_downstream_risks(
    request: dict,
    _: bool = Depends(verify_api_key)
):
    """
    Detect downstream risks from a source risk node.
    
    Args:
        request: Dictionary with source_risk_node_id
        
    Returns:
        List of downstream risk alerts
    """
    try:
        if "source_risk_node_id" not in request:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: source_risk_node_id"
            )
        
        from .analyzer import vita_analyzer
        
        alerts = await vita_analyzer.detect_downstream_risks(request["source_risk_node_id"])
        
        return {
            "downstream_alerts": alerts,
            "count": len(alerts),
            "source_node_id": request["source_risk_node_id"],
            "message": f"Generated {len(alerts)} downstream risk alerts"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to detect downstream risks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to detect downstream risks: {str(e)}")

@app.post("/intelligence/leadership_digest")
async def generate_leadership_digest(
    _: bool = Depends(verify_api_key)
):
    """
    Generate a comprehensive weekly leadership digest.
    
    Returns:
        Leadership digest with key decisions, risks, and knowledge gaps
    """
    try:
        from .analyzer import vita_analyzer
        
        digest = await vita_analyzer.generate_leadership_digest()
        
        if digest:
            return {
                "digest": digest,
                "message": "Leadership digest generated successfully"
            }
        else:
            return {
                "message": "No significant signals found for leadership digest",
                "digest": None
            }
        
    except Exception as e:
        logger.error(f"Failed to generate leadership digest: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate leadership digest: {str(e)}")

@app.post("/intelligence/knowledge_supersession")
async def detect_knowledge_supersession(
    request: dict,
    _: bool = Depends(verify_api_key)
):
    """
    Detect if new content supersedes existing knowledge.
    
    Args:
        request: Dictionary with message_content and message_id
        
    Returns:
        List of supersession actions taken
    """
    try:
        required_fields = ["message_content", "message_id"]
        missing_fields = [field for field in required_fields if field not in request]
        
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {missing_fields}"
            )
        
        from .analyzer import vita_analyzer
        
        actions = await vita_analyzer.detect_knowledge_supersession(
            request["message_content"],
            request["message_id"]
        )
        
        return {
            "supersession_actions": actions,
            "count": len(actions),
            "message_id": request["message_id"],
            "message": f"Detected {len(actions)} knowledge supersessions"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to detect knowledge supersession: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to detect knowledge supersession: {str(e)}")

@app.post("/graph-query")
async def graph_query(
    request: dict,
    _: bool = Depends(verify_api_key)
):
    """
    v6.1: GraphQL-lite endpoint for structured graph queries.
    
    Accepts JSON payload with graph traversal specifications and returns precise results.
    
    Args:
        request: JSON payload with graph query specification
        
    Returns:
        Query results based on the specified traversal
    """
    try:
        logger.info("Processing GraphQL-lite query")
        
        # Validate request structure
        if not isinstance(request, dict):
            raise HTTPException(status_code=400, detail="Request must be a JSON object")
        
        # Parse query specification
        match = request.get("match", {})
        return_fields = request.get("return", ["target.name"])
        limit = request.get("limit", 50)
        
        if not match:
            raise HTTPException(status_code=400, detail="Query must include 'match' specification")
        
        # Build SQL query from JSON specification
        sql_query, params = _build_graph_sql_query(match, return_fields, limit)
        
        # Execute query
        with vita_db.get_session() as session:
            result = session.execute(sql_query, params)
            rows = result.fetchall()
            
            # Format results
            results = []
            for row in rows:
                result_obj = {}
                for i, field in enumerate(return_fields):
                    result_obj[field] = row[i]
                results.append(result_obj)
        
        logger.info(f"GraphQL-lite query returned {len(results)} results")
        
        return {
            "results": results,
            "count": len(results),
            "query": {
                "match": match,
                "return": return_fields,
                "limit": limit
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GraphQL-lite query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Graph query failed: {str(e)}")

def _build_graph_sql_query(match: dict, return_fields: list, limit: int) -> tuple:
    """
    Build SQL query from GraphQL-lite JSON specification.
    
    Args:
        match: Match specification dictionary
        return_fields: List of fields to return
        limit: Result limit
        
    Returns:
        Tuple of (sql_query, parameters)
    """
    from sqlalchemy import text
    
    # Parse match specification
    source = match.get("source", {})
    edge = match.get("edge", {})
    target = match.get("target", {})
    
    # Build JOIN query
    query_parts = []
    params = {}
    
    # Base tables
    query_parts.append("SELECT")
    
    # Build SELECT clause
    select_parts = []
    for field in return_fields:
        if field.startswith("source."):
            column = field.replace("source.", "")
            if column in ["id", "label", "name", "status"]:
                select_parts.append(f"source_node.{column} as source_{column}")
        elif field.startswith("target."):
            column = field.replace("target.", "")
            if column in ["id", "label", "name", "status"]:
                select_parts.append(f"target_node.{column} as target_{column}")
        elif field.startswith("edge."):
            column = field.replace("edge.", "")
            if column in ["id", "relationship", "created_at"]:
                select_parts.append(f"edge.{column} as edge_{column}")
    
    if not select_parts:
        select_parts = ["target_node.name as target_name"]
    
    query_parts.append(", ".join(select_parts))
    
    # FROM clause with JOINs
    query_parts.append("""
        FROM graph_edges edge
        JOIN graph_nodes source_node ON edge.source_id = source_node.id
        JOIN graph_nodes target_node ON edge.target_id = target_node.id
    """)
    
    # WHERE clause
    where_conditions = []
    
    # Source node conditions
    if source.get("label"):
        where_conditions.append("source_node.label = :source_label")
        params["source_label"] = source["label"]
    
    if source.get("name"):
        where_conditions.append("source_node.name = :source_name")
        params["source_name"] = source["name"]
    
    if source.get("status"):
        where_conditions.append("source_node.status = :source_status")
        params["source_status"] = source["status"]
    
    # Edge conditions
    if edge.get("relationship"):
        where_conditions.append("edge.relationship = :edge_relationship")
        params["edge_relationship"] = edge["relationship"]
    
    # Target node conditions
    if target.get("label"):
        where_conditions.append("target_node.label = :target_label")
        params["target_label"] = target["label"]
    
    if target.get("name"):
        where_conditions.append("target_node.name LIKE :target_name")
        params["target_name"] = f"%{target['name']}%"
    
    if target.get("status"):
        where_conditions.append("target_node.status = :target_status")
        params["target_status"] = target["status"]
    
    # Add active status filter by default
    where_conditions.append("source_node.status = 'active'")
    where_conditions.append("target_node.status = 'active'")
    
    if where_conditions:
        query_parts.append("WHERE " + " AND ".join(where_conditions))
    
    # ORDER BY and LIMIT
    query_parts.append("ORDER BY target_node.name")
    query_parts.append(f"LIMIT {min(limit, 100)}")  # Cap at 100 results
    
    sql_query = text(" ".join(query_parts))
    
    return sql_query, params

@app.post("/admin/ontology/analyze-gaps")
async def analyze_ontology_gaps(
    request: dict,
    _: bool = Depends(verify_api_key)
):
    """
    v6.1: Analyze knowledge gaps and propose ontology updates.
    
    Uses the OntologyEvolutionAgent to identify missing concepts from failed queries.
    
    Args:
        request: Dictionary with analysis parameters
        
    Returns:
        Analysis results with proposed ontology updates
    """
    try:
        from .proactive_agents import ontology_evolution_agent
        
        days = request.get("days", 7)
        
        # Validate days parameter
        if not isinstance(days, int) or days < 1 or days > 30:
            raise HTTPException(
                status_code=400,
                detail="Days parameter must be an integer between 1 and 30"
            )
        
        logger.info(f"Starting ontology gap analysis for last {days} days")
        
        # Run the analysis
        analysis_results = await ontology_evolution_agent.analyze_knowledge_gaps(days)
        
        # Update metrics after analysis
        update_metrics()
        
        return {
            "analysis": analysis_results,
            "message": f"Ontology gap analysis completed for last {days} days",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze ontology gaps: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze ontology gaps: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", 8000))
    
    uvicorn.run(app, host=host, port=port, log_level="info") 