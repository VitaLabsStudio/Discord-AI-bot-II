#!/usr/bin/env python3
"""
VITA Data Integrity Verification Script
=======================================

This script performs a comprehensive audit of VITA's data persistence layer
by comparing Discord ground truth with stored data in SQLite and Pinecone.

Usage:
    python verify_data_integrity.py <message_id>
    
Example:
    python verify_data_integrity.py 1234567890123456789
"""

import os
import sys
import json
import hashlib
import sqlite3
import asyncio
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Discord API imports
import discord
from discord.ext import commands

# Pinecone imports
from pinecone import Pinecone

# Import VITA utilities
sys.path.append('src')
from backend.utils import clean_text, redact_sensitive_info, split_text_for_embedding
from backend.ingestion import compute_content_hash

# Load environment variables
load_dotenv()

class DataIntegrityVerifier:
    """Comprehensive data integrity verification for VITA system."""
    
    def __init__(self):
        self.discord_token = os.getenv("DISCORD_TOKEN")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "vita-knowledge-base")
        self.db_path = "vita_data.db"
        
        # Initialize connections
        self.bot = None
        self.pinecone_client = None
        self.pinecone_index = None
        
        # Verification results
        self.results = {
            "message_id": None,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
            "overall_status": "UNKNOWN",
            "errors": [],
            "warnings": []
        }
    
    async def initialize(self):
        """Initialize Discord bot and Pinecone connections."""
        try:
            # Initialize Discord bot
            intents = discord.Intents.default()
            intents.message_content = True
            self.bot = commands.Bot(command_prefix='!', intents=intents)
            
            # Login to Discord
            await self.bot.login(self.discord_token)
            
            # Initialize Pinecone
            self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
            self.pinecone_index = self.pinecone_client.Index(self.pinecone_index_name)
            
            print("✅ [INIT] Successfully initialized Discord and Pinecone connections")
            
        except Exception as e:
            print(f"❌ [INIT] Failed to initialize connections: {e}")
            raise
    
    async def cleanup(self):
        """Clean up connections."""
        if self.bot:
            await self.bot.close()
    
    async def verify_message(self, message_id: str) -> Dict[str, Any]:
        """
        Perform comprehensive verification of a message across all data stores.
        
        Args:
            message_id: Discord message ID to verify
            
        Returns:
            Dictionary with verification results
        """
        self.results["message_id"] = message_id
        print(f"\n🔍 Starting comprehensive verification for message: {message_id}")
        print("=" * 80)
        
        try:
            # Step 1: Fetch ground truth from Discord
            discord_data = await self._fetch_discord_ground_truth(message_id)
            if not discord_data:
                self.results["overall_status"] = "FAILED"
                return self.results
            
            # Step 2: Verify SQLite processed_messages table
            await self._verify_sqlite_processed_messages(message_id, discord_data)
            
            # Step 3: Verify SQLite attachments table
            await self._verify_sqlite_attachments(message_id, discord_data)
            
            # Step 4: Verify Pinecone vectors
            await self._verify_pinecone_vectors(message_id, discord_data)
            
            # Step 5: Verify Knowledge Graph entries
            await self._verify_knowledge_graph(message_id, discord_data)
            
            # Step 6: Cross-reference consistency
            await self._verify_cross_system_consistency(message_id, discord_data)
            
            # Determine overall status
            self._calculate_overall_status()
            
        except Exception as e:
            self.results["errors"].append(f"Critical verification error: {e}")
            self.results["overall_status"] = "ERROR"
        
        return self.results
    
    async def _fetch_discord_ground_truth(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Fetch message data directly from Discord API."""
        print("\n📡 Step 1: Fetching ground truth from Discord")
        print("-" * 50)
        
        try:
            # Search across all accessible channels for the message
            # This is a brute force approach but necessary since we don't know the channel
            message = None
            channel_found = None
            
            for guild in self.bot.guilds:
                for channel in guild.text_channels:
                    try:
                        message = await channel.fetch_message(int(message_id))
                        channel_found = channel
                        break
                    except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                        continue
                if message:
                    break
            
            if not message:
                print(f"❌ [DISCORD] Message {message_id} not found in any accessible channel")
                self.results["checks"]["discord_fetch"] = {
                    "status": "FAIL",
                    "error": "Message not found in Discord"
                }
                return None
            
            # Extract all relevant data
            discord_data = {
                "message_id": str(message.id),
                "channel_id": str(message.channel.id),
                "channel_name": message.channel.name,
                "guild_id": str(message.guild.id) if message.guild else None,
                "guild_name": message.guild.name if message.guild else None,
                "author_id": str(message.author.id),
                "author_name": str(message.author),
                "content": message.content,
                "timestamp": message.created_at.isoformat(),
                "edited_at": message.edited_at.isoformat() if message.edited_at else None,
                "attachments": [],
                "thread_id": str(message.thread.id) if hasattr(message, 'thread') and message.thread else None,
                "is_pinned": message.pinned,
                "message_type": str(message.type)
            }
            
            # Process attachments
            for attachment in message.attachments:
                attachment_data = {
                    "id": str(attachment.id),
                    "filename": attachment.filename,
                    "size": attachment.size,
                    "url": attachment.url,
                    "content_type": attachment.content_type,
                    "width": getattr(attachment, 'width', None),
                    "height": getattr(attachment, 'height', None)
                }
                
                # Compute attachment hash (simplified - in real system we'd download and hash)
                attachment_hash = hashlib.sha256(f"{attachment.url}{attachment.size}".encode()).hexdigest()
                attachment_data["computed_hash"] = attachment_hash
                
                discord_data["attachments"].append(attachment_data)
            
            # Compute content hash using VITA's algorithm
            all_content = []
            if discord_data["content"] and discord_data["content"].strip():
                cleaned_content = clean_text(discord_data["content"])
                redacted_content = redact_sensitive_info(cleaned_content)
                if redacted_content:
                    all_content.append(redacted_content)
            
            # Add attachment hashes for content hash computation
            attachment_ids = [att["computed_hash"] for att in discord_data["attachments"]]
            combined_content = "\n\n".join(all_content) if all_content else ""
            discord_data["computed_content_hash"] = compute_content_hash(combined_content, attachment_ids)
            discord_data["combined_content"] = combined_content
            
            print(f"✅ [DISCORD] Successfully fetched message from #{channel_found.name}")
            print(f"   Author: {discord_data['author_name']}")
            print(f"   Content length: {len(discord_data['content'])} chars")
            print(f"   Attachments: {len(discord_data['attachments'])}")
            print(f"   Computed content hash: {discord_data['computed_content_hash'][:16]}...")
            
            self.results["checks"]["discord_fetch"] = {
                "status": "PASS",
                "channel_name": channel_found.name,
                "author": discord_data['author_name'],
                "content_length": len(discord_data['content']),
                "attachment_count": len(discord_data['attachments']),
                "content_hash": discord_data['computed_content_hash'][:16] + "..."
            }
            
            return discord_data
            
        except Exception as e:
            print(f"❌ [DISCORD] Failed to fetch message: {e}")
            self.results["checks"]["discord_fetch"] = {
                "status": "ERROR",
                "error": str(e)
            }
            return None
    
    async def _verify_sqlite_processed_messages(self, message_id: str, discord_data: Dict):
        """Verify message exists in SQLite processed_messages table."""
        print("\n🗄️  Step 2: Verifying SQLite processed_messages table")
        print("-" * 50)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query for the message
            cursor.execute("""
                SELECT message_id, timestamp, channel_id, user_id, content_hash
                FROM processed_messages 
                WHERE message_id = ?
            """, (message_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                print(f"❌ [SQLITE] Message {message_id} not found in processed_messages table")
                self.results["checks"]["sqlite_processed_messages"] = {
                    "status": "FAIL",
                    "error": "Message not found in processed_messages table"
                }
                return
            
            db_message_id, db_timestamp, db_channel_id, db_user_id, db_content_hash = row
            
            # Verify data consistency
            checks = []
            
            # Check message_id
            if db_message_id == message_id:
                checks.append("✅ Message ID matches")
            else:
                checks.append(f"❌ Message ID mismatch: DB={db_message_id}, Expected={message_id}")
            
            # Check channel_id
            if db_channel_id == discord_data["channel_id"]:
                checks.append("✅ Channel ID matches")
            else:
                checks.append(f"❌ Channel ID mismatch: DB={db_channel_id}, Discord={discord_data['channel_id']}")
            
            # Check user_id
            if db_user_id == discord_data["author_id"]:
                checks.append("✅ User ID matches")
            else:
                checks.append(f"❌ User ID mismatch: DB={db_user_id}, Discord={discord_data['author_id']}")
            
            # Check content_hash
            if db_content_hash == discord_data["computed_content_hash"]:
                checks.append("✅ Content hash matches")
            elif db_content_hash is None:
                checks.append("⚠️  Content hash is NULL (legacy entry)")
                self.results["warnings"].append("Content hash is NULL - may be legacy entry")
            else:
                checks.append(f"❌ Content hash mismatch: DB={db_content_hash[:16]}..., Computed={discord_data['computed_content_hash'][:16]}...")
            
            # Print results
            for check in checks:
                print(f"   {check}")
            
            # Determine status
            failed_checks = [c for c in checks if c.startswith("❌")]
            if failed_checks:
                status = "FAIL"
            elif any(c.startswith("⚠️") for c in checks):
                status = "WARN"
            else:
                status = "PASS"
            
            self.results["checks"]["sqlite_processed_messages"] = {
                "status": status,
                "checks": checks,
                "db_content_hash": db_content_hash[:16] + "..." if db_content_hash else None,
                "computed_content_hash": discord_data["computed_content_hash"][:16] + "..."
            }
            
        except Exception as e:
            print(f"❌ [SQLITE] Error verifying processed_messages: {e}")
            self.results["checks"]["sqlite_processed_messages"] = {
                "status": "ERROR",
                "error": str(e)
            }
    
    async def _verify_sqlite_attachments(self, message_id: str, discord_data: Dict):
        """Verify attachments exist in SQLite attachments table."""
        print("\n📎 Step 3: Verifying SQLite attachments table")
        print("-" * 50)
        
        if not discord_data["attachments"]:
            print("   ℹ️  No attachments to verify")
            self.results["checks"]["sqlite_attachments"] = {
                "status": "PASS",
                "message": "No attachments to verify"
            }
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            checks = []
            found_attachments = 0
            
            for attachment in discord_data["attachments"]:
                attachment_hash = attachment["computed_hash"]
                
                # Query for the attachment
                cursor.execute("""
                    SELECT attachment_id, original_filename, file_size_bytes, mime_type
                    FROM attachments 
                    WHERE attachment_id = ?
                """, (attachment_hash,))
                
                row = cursor.fetchone()
                
                if row:
                    db_id, db_filename, db_size, db_mime = row
                    
                    # Verify attachment data
                    if db_filename == attachment["filename"]:
                        checks.append(f"✅ Attachment {attachment['filename']} found with correct filename")
                    else:
                        checks.append(f"❌ Attachment filename mismatch: DB={db_filename}, Discord={attachment['filename']}")
                    
                    if db_size == attachment["size"]:
                        checks.append(f"✅ Attachment size matches ({db_size} bytes)")
                    else:
                        checks.append(f"❌ Attachment size mismatch: DB={db_size}, Discord={attachment['size']}")
                    
                    found_attachments += 1
                else:
                    checks.append(f"❌ Attachment {attachment['filename']} not found in database")
                    checks.append(f"   Expected hash: {attachment_hash[:16]}...")
            
            conn.close()
            
            # Print results
            for check in checks:
                print(f"   {check}")
            
            # Determine status
            expected_count = len(discord_data["attachments"])
            if found_attachments == expected_count:
                status = "PASS"
            elif found_attachments > 0:
                status = "PARTIAL"
            else:
                status = "FAIL"
            
            self.results["checks"]["sqlite_attachments"] = {
                "status": status,
                "expected_count": expected_count,
                "found_count": found_attachments,
                "checks": checks
            }
            
        except Exception as e:
            print(f"❌ [SQLITE] Error verifying attachments: {e}")
            self.results["checks"]["sqlite_attachments"] = {
                "status": "ERROR",
                "error": str(e)
            }
    
    async def _verify_pinecone_vectors(self, message_id: str, discord_data: Dict):
        """Verify vectors exist in Pinecone with correct metadata."""
        print("\n🎯 Step 4: Verifying Pinecone vectors")
        print("-" * 50)
        
        try:
            # Query Pinecone for vectors with this message_id
            query_response = self.pinecone_index.query(
                vector=[0.0] * 1536,  # Dummy vector for metadata-only query
                top_k=100,  # Get up to 100 chunks
                include_metadata=True,
                filter={"message_id": message_id}
            )
            
            matches = query_response.get('matches', [])
            
            if not matches:
                print(f"❌ [PINECONE] No vectors found for message {message_id}")
                self.results["checks"]["pinecone_vectors"] = {
                    "status": "FAIL",
                    "error": "No vectors found in Pinecone"
                }
                return
            
            print(f"✅ [PINECONE] Found {len(matches)} vector(s) for message {message_id}")
            
            # Verify metadata consistency
            checks = []
            metadata_issues = []
            
            # Calculate expected chunk count
            if discord_data["combined_content"]:
                expected_chunks = len(split_text_for_embedding(discord_data["combined_content"]))
            else:
                expected_chunks = 0
            
            # Check chunk count
            if len(matches) == expected_chunks:
                checks.append(f"✅ Chunk count matches expected ({len(matches)})")
            else:
                checks.append(f"⚠️  Chunk count mismatch: Found={len(matches)}, Expected={expected_chunks}")
                self.results["warnings"].append(f"Pinecone chunk count mismatch for {message_id}")
            
            # Verify metadata for each vector
            for i, match in enumerate(matches):
                metadata = match.get('metadata', {})
                
                # Check required fields
                required_fields = ['message_id', 'channel_id', 'user_id', 'timestamp', 'content']
                for field in required_fields:
                    if field not in metadata:
                        metadata_issues.append(f"Missing field '{field}' in vector {i}")
                    elif field == 'message_id' and metadata[field] != message_id:
                        metadata_issues.append(f"Message ID mismatch in vector {i}: {metadata[field]} != {message_id}")
                    elif field == 'channel_id' and metadata[field] != discord_data['channel_id']:
                        metadata_issues.append(f"Channel ID mismatch in vector {i}: {metadata[field]} != {discord_data['channel_id']}")
                    elif field == 'user_id' and metadata[field] != discord_data['author_id']:
                        metadata_issues.append(f"User ID mismatch in vector {i}: {metadata[field]} != {discord_data['author_id']}")
                
                # Check content hash if present
                if 'content_hash' in metadata:
                    if metadata['content_hash'] == discord_data['computed_content_hash']:
                        checks.append(f"✅ Vector {i} content hash matches")
                    else:
                        metadata_issues.append(f"Content hash mismatch in vector {i}")
                
                # Check attachment IDs if present
                if discord_data["attachments"] and 'attachment_ids' in metadata:
                    expected_attachment_ids = [att["computed_hash"] for att in discord_data["attachments"]]
                    # Note: attachment_ids might be stored as string representation of list
                    stored_attachment_ids = metadata['attachment_ids']
                    if isinstance(stored_attachment_ids, str):
                        try:
                            stored_attachment_ids = eval(stored_attachment_ids)
                        except:
                            pass
                    
                    if set(stored_attachment_ids) == set(expected_attachment_ids):
                        checks.append(f"✅ Vector {i} attachment IDs match")
                    else:
                        metadata_issues.append(f"Attachment IDs mismatch in vector {i}")
            
            # Print results
            for check in checks:
                print(f"   {check}")
            
            if metadata_issues:
                print("   Metadata issues found:")
                for issue in metadata_issues[:10]:  # Limit to first 10 issues
                    print(f"   ❌ {issue}")
                if len(metadata_issues) > 10:
                    print(f"   ... and {len(metadata_issues) - 10} more issues")
            
            # Determine status
            if metadata_issues:
                status = "FAIL" if len(metadata_issues) > len(matches) * 0.5 else "WARN"
            else:
                status = "PASS"
            
            self.results["checks"]["pinecone_vectors"] = {
                "status": status,
                "vector_count": len(matches),
                "expected_chunks": expected_chunks,
                "metadata_issues": len(metadata_issues),
                "checks": checks,
                "sample_metadata": matches[0].get('metadata', {}) if matches else {}
            }
            
        except Exception as e:
            print(f"❌ [PINECONE] Error verifying vectors: {e}")
            self.results["checks"]["pinecone_vectors"] = {
                "status": "ERROR",
                "error": str(e)
            }
    
    async def _verify_knowledge_graph(self, message_id: str, discord_data: Dict):
        """Verify knowledge graph entries related to the message."""
        print("\n🕸️  Step 5: Verifying Knowledge Graph entries")
        print("-" * 50)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query for nodes that might be related to this message
            cursor.execute("""
                SELECT COUNT(*) FROM graph_nodes 
                WHERE properties LIKE ?
            """, (f'%{message_id}%',))
            
            node_count = cursor.fetchone()[0]
            
            # Query for edges that reference this message
            cursor.execute("""
                SELECT COUNT(*) FROM graph_edges 
                WHERE message_id = ?
            """, (message_id,))
            
            edge_count = cursor.fetchone()[0]
            
            conn.close()
            
            if node_count == 0 and edge_count == 0:
                print("   ℹ️  No knowledge graph entries found (this is normal for most messages)")
                status = "PASS"
                message = "No knowledge graph entries (normal)"
            else:
                print(f"   ✅ Found {node_count} related nodes and {edge_count} edges")
                status = "PASS"
                message = f"Found {node_count} nodes and {edge_count} edges"
            
            self.results["checks"]["knowledge_graph"] = {
                "status": status,
                "node_count": node_count,
                "edge_count": edge_count,
                "message": message
            }
            
        except Exception as e:
            print(f"❌ [GRAPH] Error verifying knowledge graph: {e}")
            self.results["checks"]["knowledge_graph"] = {
                "status": "ERROR",
                "error": str(e)
            }
    
    async def _verify_cross_system_consistency(self, message_id: str, discord_data: Dict):
        """Verify consistency across all systems."""
        print("\n🔗 Step 6: Verifying cross-system consistency")
        print("-" * 50)
        
        checks = []
        issues = []
        
        # Check if message exists in SQLite but not Pinecone (or vice versa)
        sqlite_status = self.results["checks"].get("sqlite_processed_messages", {}).get("status")
        pinecone_status = self.results["checks"].get("pinecone_vectors", {}).get("status")
        
        if sqlite_status in ["PASS", "WARN"] and pinecone_status in ["PASS", "WARN"]:
            checks.append("✅ Message exists in both SQLite and Pinecone")
        elif sqlite_status in ["PASS", "WARN"] and pinecone_status == "FAIL":
            issues.append("Message exists in SQLite but not in Pinecone")
        elif sqlite_status == "FAIL" and pinecone_status in ["PASS", "WARN"]:
            issues.append("Message exists in Pinecone but not in SQLite")
        else:
            issues.append("Message missing from both SQLite and Pinecone")
        
        # Check attachment consistency
        sqlite_att_status = self.results["checks"].get("sqlite_attachments", {}).get("status")
        if discord_data["attachments"]:
            if sqlite_att_status in ["PASS", "PARTIAL"]:
                checks.append("✅ Attachments properly registered")
            else:
                issues.append("Attachments missing from SQLite")
        
        # Check content hash consistency
        sqlite_hash = self.results["checks"].get("sqlite_processed_messages", {}).get("db_content_hash", "").replace("...", "")
        computed_hash = discord_data["computed_content_hash"][:16]
        
        if sqlite_hash and sqlite_hash == computed_hash:
            checks.append("✅ Content hashes consistent across systems")
        elif not sqlite_hash:
            checks.append("⚠️  Content hash missing in SQLite (legacy entry)")
        else:
            issues.append("Content hash inconsistency detected")
        
        # Print results
        for check in checks:
            print(f"   {check}")
        
        if issues:
            print("   Issues found:")
            for issue in issues:
                print(f"   ❌ {issue}")
        
        # Determine status
        if issues:
            status = "FAIL"
        elif any("⚠️" in check for check in checks):
            status = "WARN"
        else:
            status = "PASS"
        
        self.results["checks"]["cross_system_consistency"] = {
            "status": status,
            "checks": checks,
            "issues": issues
        }
    
    def _calculate_overall_status(self):
        """Calculate overall verification status."""
        statuses = [check.get("status", "UNKNOWN") for check in self.results["checks"].values()]
        
        if "ERROR" in statuses:
            self.results["overall_status"] = "ERROR"
        elif "FAIL" in statuses:
            self.results["overall_status"] = "FAILED"
        elif "WARN" in statuses:
            self.results["overall_status"] = "WARNING"
        elif all(status == "PASS" for status in statuses):
            self.results["overall_status"] = "PASSED"
        else:
            self.results["overall_status"] = "UNKNOWN"
    
    def print_summary(self):
        """Print a comprehensive summary of verification results."""
        print("\n" + "=" * 80)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 80)
        
        overall_status = self.results["overall_status"]
        status_emoji = {
            "PASSED": "✅",
            "WARNING": "⚠️",
            "FAILED": "❌",
            "ERROR": "💥",
            "UNKNOWN": "❓"
        }
        
        print(f"\n{status_emoji.get(overall_status, '❓')} Overall Status: {overall_status}")
        print(f"📝 Message ID: {self.results['message_id']}")
        print(f"🕒 Verification Time: {self.results['timestamp']}")
        
        print(f"\n📋 Individual Check Results:")
        print("-" * 40)
        
        for check_name, check_data in self.results["checks"].items():
            status = check_data.get("status", "UNKNOWN")
            emoji = status_emoji.get(status, "❓")
            check_display = check_name.replace("_", " ").title()
            print(f"{emoji} {check_display}: {status}")
            
            # Show additional details for failed checks
            if status in ["FAIL", "ERROR"] and "error" in check_data:
                print(f"   Error: {check_data['error']}")
        
        # Show warnings and errors
        if self.results["warnings"]:
            print(f"\n⚠️  Warnings ({len(self.results['warnings'])}):")
            for warning in self.results["warnings"]:
                print(f"   • {warning}")
        
        if self.results["errors"]:
            print(f"\n❌ Errors ({len(self.results['errors'])}):")
            for error in self.results["errors"]:
                print(f"   • {error}")
        
        print("\n" + "=" * 80)

async def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(
        description="Verify data integrity for a Discord message across VITA's data stores"
    )
    parser.add_argument("message_id", help="Discord message ID to verify")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not args.message_id.isdigit():
        print("❌ Error: Message ID must be a numeric Discord message ID")
        sys.exit(1)
    
    verifier = DataIntegrityVerifier()
    
    try:
        await verifier.initialize()
        results = await verifier.verify_message(args.message_id)
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            verifier.print_summary()
        
        # Exit with appropriate code
        exit_codes = {
            "PASSED": 0,
            "WARNING": 1,
            "FAILED": 2,
            "ERROR": 3,
            "UNKNOWN": 4
        }
        sys.exit(exit_codes.get(results["overall_status"], 4))
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Verification interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Critical error during verification: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        await verifier.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 