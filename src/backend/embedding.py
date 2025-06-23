import os
import uuid
import asyncio
from typing import List, Dict, Any
from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from .logger import get_logger
from .utils import sanitize_metadata
from .retry_utils import retry_on_api_error

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

class EmbeddingManager:
    """Manages OpenAI embeddings and Pinecone vector storage."""
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
        # Initialize Pinecone
        self.pinecone_client = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY")
        )
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "vita-knowledge-base")
        
        # Initialize index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize Pinecone index if it doesn't exist."""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=1536,  # text-embedding-3-small dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            
            self.index = self.pinecone_client.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {e}")
            raise
    
    @retry_on_api_error(max_attempts=3, min_wait=1.0, max_wait=30.0)
    async def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text chunks using OpenAI with retry logic.
        
        Args:
            chunks: List of text chunks to embed
            
        Returns:
            List of embedding vectors
        """
        if not chunks:
            return []
        
        try:
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            
            # Process chunks in batches to avoid rate limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                response = await self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                if i + batch_size < len(chunks):
                    await asyncio.sleep(0.1)
            
            logger.info(f"Generated {len(all_embeddings)} embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def store_embeddings(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """
        Store embeddings in Pinecone with metadata.
        
        Args:
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
        """
        if not embeddings or not metadatas:
            logger.warning("No embeddings or metadata to store")
            return
        
        if len(embeddings) != len(metadatas):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        try:
            logger.info(f"Storing {len(embeddings)} embeddings in Pinecone")
            
            # CRITICAL FIX: Use UUID for vector IDs, store message_id in metadata
            vectors = []
            for embedding, metadata in zip(embeddings, metadatas):
                vector_id = str(uuid.uuid4())
                sanitized_metadata = sanitize_metadata(metadata)
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": sanitized_metadata
                })
            
            # Store in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.debug(f"Stored batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            
            logger.info(f"Successfully stored {len(vectors)} vectors")
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            raise
    
    async def query_similar(self, query_text: str, top_k: int = 5, filter_dict: Dict = None) -> List[Dict]:
        """
        Query similar vectors from Pinecone.
        
        Args:
            query_text: Text to query for
            top_k: Number of results to return
            filter_dict: Pinecone filter dictionary
            
        Returns:
            List of similar documents with metadata
        """
        try:
            # Generate embedding for query
            query_embedding = await self.embed_chunks([query_text])
            if not query_embedding:
                return []
            
            # Query Pinecone
            query_response = self.index.query(
                vector=query_embedding[0],
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            results = []
            for match in query_response.matches:
                result = {
                    "score": match.score,
                    "metadata": match.metadata
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Failed to query similar documents: {e}")
            return []

# Global embedding manager instance
embedding_manager = EmbeddingManager() 