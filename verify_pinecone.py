import os
import argparse
from dotenv import load_dotenv
from pinecone import Pinecone

def verify_message_in_pinecone(message_id: str):
    """
    Connects to Pinecone and fetches all vectors associated with a given message_id.
    """
    print(f"--- Verifying Message ID: {message_id} in Pinecone ---")

    # Load environment variables from .env file
    load_dotenv()

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    if not all([pinecone_api_key, pinecone_index_name]):
        print("Error: Ensure PINECONE_API_KEY and PINECONE_INDEX_NAME are set in your .env file.")
        return

    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)
        print(f"Successfully connected to index '{pinecone_index_name}'.")
        print(f"Index stats: {index.describe_index_stats()}")

        # Pinecone's fetch method can retrieve by ID, but we stored the message_id in metadata.
        # We need to perform a dummy query and filter by metadata to find our vectors.
        # Note: This requires a vector, so we create a dummy one. The actual vector values don't matter for this check.
        dummy_vector = [0.0] * 1536 # Dimension for text-embedding-3-small

        query_response = index.query(
            vector=dummy_vector,
            top_k=10,  # Retrieve up to 10 chunks for this message
            include_metadata=True,
            filter={
                "message_id": {"$eq": message_id}
            }
        )

        matches = query_response.get('matches', [])

        if not matches:
            print(f"\nRESULT: NOT FOUND. No vectors found for message_id '{message_id}'.")
            return

        print(f"\nRESULT: SUCCESS! Found {len(matches)} vector(s) for message_id '{message_id}'.")

        for i, match in enumerate(matches):
            metadata = match.get('metadata', {})
            print(f"\n--- Chunk {i+1} ---")
            print(f"  Vector ID: {match.get('id')}")
            print(f"  Score: {match.get('score')}")
            print(f"  Metadata:")
            for key, value in metadata.items():
                # Truncate long content for readability
                if key == 'content' and isinstance(value, str) and len(value) > 200:
                    print(f"    {key}: '{value[:200]}...'")
                else:
                    print(f"    {key}: {value}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a message's ingestion in Pinecone.")
    parser.add_argument("message_id", type=str, help="The Discord message ID to verify.")
    args = parser.parse_args()

    verify_message_in_pinecone(args.message_id) 