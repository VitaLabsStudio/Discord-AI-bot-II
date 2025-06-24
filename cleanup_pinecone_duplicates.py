#!/usr/bin/env python3
"""
Pinecone Duplicate Cleanup Script
=================================

This script identifies and removes duplicate vectors from Pinecone that are causing
the same documents to be returned repeatedly in search results.

It will:
1. Query Pinecone for vectors with the same message_id
2. Keep only the first vector for each unique message_id
3. Delete duplicate vectors
4. Improve search result diversity
"""

import os
import sys
from collections import defaultdict
from dotenv import load_dotenv
from pinecone import Pinecone

# Add src to path for imports
sys.path.append('src')

def cleanup_pinecone_duplicates():
    """Remove duplicate vectors from Pinecone to improve search diversity."""
    
    load_dotenv()
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index('curoseoaiembed')
    
    print("🔍 Starting Pinecone duplicate cleanup...")
    
    # Get index stats
    stats = index.describe_index_stats()
    print(f"📊 Total vectors in index: {stats.total_vector_count}")
    
    # Fetch all vector IDs and metadata
    print("📥 Fetching all vector metadata...")
    
    # Use list_paginated to get all vector IDs
    vector_ids = []
    for ids_batch in index.list_paginated():
        vector_ids.extend(ids_batch)
    
    print(f"📋 Found {len(vector_ids)} vectors")
    
    # Group vectors by message_id
    message_vectors = defaultdict(list)
    
    # Fetch metadata in batches
    batch_size = 100
    for i in range(0, len(vector_ids), batch_size):
        batch_ids = vector_ids[i:i+batch_size]
        
        try:
            fetch_result = index.fetch(ids=batch_ids)
            
            for vector_id, vector_data in fetch_result.vectors.items():
                if vector_data.metadata and 'message_id' in vector_data.metadata:
                    message_id = vector_data.metadata['message_id']
                    message_vectors[message_id].append({
                        'id': vector_id,
                        'metadata': vector_data.metadata
                    })
                else:
                    print(f"⚠️  Vector {vector_id} has no message_id metadata")
        
        except Exception as e:
            print(f"❌ Error fetching batch {i//batch_size + 1}: {e}")
            continue
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"📊 Processed {i + len(batch_ids)} vectors...")
    
    # Identify duplicates
    duplicates_to_delete = []
    messages_with_duplicates = 0
    
    for message_id, vectors in message_vectors.items():
        if len(vectors) > 1:
            messages_with_duplicates += 1
            # Keep the first vector, mark others for deletion
            vectors_to_keep = vectors[:1]
            vectors_to_delete = vectors[1:]
            
            print(f"🔍 Message {message_id}: {len(vectors)} vectors, keeping 1, deleting {len(vectors_to_delete)}")
            
            for vector in vectors_to_delete:
                duplicates_to_delete.append(vector['id'])
    
    print(f"\n📈 Summary:")
    print(f"   • Total messages: {len(message_vectors)}")
    print(f"   • Messages with duplicates: {messages_with_duplicates}")
    print(f"   • Duplicate vectors to delete: {len(duplicates_to_delete)}")
    
    if not duplicates_to_delete:
        print("✅ No duplicates found! Index is clean.")
        return
    
    # Confirm deletion
    response = input(f"\n❓ Delete {len(duplicates_to_delete)} duplicate vectors? (y/N): ")
    if response.lower() != 'y':
        print("❌ Cleanup cancelled.")
        return
    
    # Delete duplicates in batches
    print(f"🗑️  Deleting {len(duplicates_to_delete)} duplicate vectors...")
    
    delete_batch_size = 100
    deleted_count = 0
    
    for i in range(0, len(duplicates_to_delete), delete_batch_size):
        batch_ids = duplicates_to_delete[i:i+delete_batch_size]
        
        try:
            index.delete(ids=batch_ids)
            deleted_count += len(batch_ids)
            print(f"🗑️  Deleted batch {i//delete_batch_size + 1}: {len(batch_ids)} vectors")
        except Exception as e:
            print(f"❌ Error deleting batch {i//delete_batch_size + 1}: {e}")
    
    print(f"\n✅ Cleanup complete!")
    print(f"   • Deleted {deleted_count} duplicate vectors")
    print(f"   • Remaining vectors: {stats.total_vector_count - deleted_count}")
    
    # Verify cleanup
    final_stats = index.describe_index_stats()
    print(f"   • Final vector count: {final_stats.total_vector_count}")

def identify_content_issues():
    """Identify messages with problematic content that might be causing poor search results."""
    
    import sqlite3
    
    print("\n🔍 Analyzing message content for search issues...")
    
    conn = sqlite3.connect('vita_data.db')
    cursor = conn.cursor()
    
    # Check for very short content
    cursor.execute("""
        SELECT message_id, content, LENGTH(content) as content_length
        FROM processed_messages 
        WHERE content IS NOT NULL AND LENGTH(content) < 50
        ORDER BY content_length
        LIMIT 10
    """)
    
    short_messages = cursor.fetchall()
    if short_messages:
        print(f"\n⚠️  Found {len(short_messages)} messages with very short content:")
        for msg_id, content, length in short_messages:
            print(f"   • {msg_id}: '{content[:30]}...' ({length} chars)")
    
    # Check for duplicate content
    cursor.execute("""
        SELECT content, COUNT(*) as count
        FROM processed_messages 
        WHERE content IS NOT NULL
        GROUP BY content
        HAVING COUNT(*) > 1
        ORDER BY count DESC
        LIMIT 10
    """)
    
    duplicate_content = cursor.fetchall()
    if duplicate_content:
        print(f"\n🔄 Found {len(duplicate_content)} content duplicates:")
        for content, count in duplicate_content:
            print(f"   • '{content[:50]}...' appears {count} times")
    
    # Check content distribution
    cursor.execute("""
        SELECT 
            COUNT(*) as total_messages,
            COUNT(CASE WHEN content IS NOT NULL THEN 1 END) as with_content,
            AVG(LENGTH(content)) as avg_length
        FROM processed_messages
    """)
    
    stats = cursor.fetchone()
    total, with_content, avg_length = stats
    print(f"\n📊 Content Statistics:")
    print(f"   • Total messages: {total}")
    print(f"   • Messages with content: {with_content} ({with_content/total*100:.1f}%)")
    print(f"   • Average content length: {avg_length:.0f} characters")
    
    conn.close()

if __name__ == "__main__":
    print("🚀 VITA Pinecone Cleanup Tool")
    print("=" * 40)
    
    try:
        # First analyze content issues
        identify_content_issues()
        
        # Then clean up Pinecone duplicates
        cleanup_pinecone_duplicates()
        
        print("\n🎉 All cleanup tasks completed!")
        print("\n💡 Next steps:")
        print("   1. Test search queries to verify improved diversity")
        print("   2. Run a new ingestion to populate content in database")
        print("   3. Monitor search results for better variety")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        sys.exit(1) 