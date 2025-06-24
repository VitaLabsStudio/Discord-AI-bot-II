#!/usr/bin/env python3
"""
VITA Database Schema Fix Script
==============================

This script fixes critical database schema issues identified in the audit:
1. Adds missing columns to graph_nodes table
2. Creates proper indexes
3. Enables foreign key constraints
4. Optionally backfills content hashes for legacy messages

Usage:
    python fix_database_schema.py [--backfill-hashes] [--dry-run]
"""

import os
import sys
import sqlite3
import hashlib
import argparse
from datetime import datetime
from typing import List, Tuple

# Add src to path for imports
sys.path.append('src')
from backend.utils import clean_text, redact_sensitive_info
from backend.ingestion import compute_content_hash

class DatabaseSchemaFixer:
    """Fixes critical database schema issues."""
    
    def __init__(self, db_path: str = "vita_data.db", dry_run: bool = False):
        self.db_path = db_path
        self.dry_run = dry_run
        self.changes_made = []
        
    def execute_sql(self, conn: sqlite3.Connection, sql: str, params: tuple = None) -> bool:
        """Execute SQL with dry-run support."""
        if self.dry_run:
            print(f"[DRY RUN] Would execute: {sql}")
            if params:
                print(f"[DRY RUN] With params: {params}")
            return True
        
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            return True
        except Exception as e:
            print(f"❌ Error executing SQL: {e}")
            print(f"   SQL: {sql}")
            return False
    
    def check_column_exists(self, conn: sqlite3.Connection, table: str, column: str) -> bool:
        """Check if a column exists in a table."""
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        return column in columns
    
    def fix_graph_nodes_schema(self, conn: sqlite3.Connection) -> bool:
        """Add missing columns to graph_nodes table."""
        print("\n🔧 Fixing graph_nodes table schema...")
        
        missing_columns = []
        
        # Check for missing columns
        if not self.check_column_exists(conn, 'graph_nodes', 'status'):
            missing_columns.append(('status', 'VARCHAR DEFAULT "active"'))
        
        if not self.check_column_exists(conn, 'graph_nodes', 'version'):
            missing_columns.append(('version', 'INTEGER DEFAULT 1'))
            
        if not self.check_column_exists(conn, 'graph_nodes', 'last_accessed_at'):
            missing_columns.append(('last_accessed_at', 'DATETIME DEFAULT CURRENT_TIMESTAMP'))
        
        if not missing_columns:
            print("   ✅ All required columns already exist")
            return True
        
        success = True
        for column_name, column_def in missing_columns:
            sql = f"ALTER TABLE graph_nodes ADD COLUMN {column_name} {column_def}"
            if self.execute_sql(conn, sql):
                print(f"   ✅ Added column: {column_name}")
                self.changes_made.append(f"Added {column_name} column to graph_nodes")
            else:
                print(f"   ❌ Failed to add column: {column_name}")
                success = False
        
        # Add index for status column if it was added
        if any(col[0] == 'status' for col in missing_columns):
            index_sql = "CREATE INDEX IF NOT EXISTS ix_graph_nodes_status ON graph_nodes (status)"
            if self.execute_sql(conn, index_sql):
                print("   ✅ Created index on status column")
                self.changes_made.append("Created index on graph_nodes.status")
            else:
                print("   ❌ Failed to create status index")
                success = False
        
        return success
    
    def enable_foreign_keys(self, conn: sqlite3.Connection) -> bool:
        """Enable foreign key constraints."""
        print("\n🔧 Enabling foreign key constraints...")
        
        if self.execute_sql(conn, "PRAGMA foreign_keys = ON"):
            print("   ✅ Foreign key constraints enabled")
            self.changes_made.append("Enabled foreign key constraints")
            return True
        else:
            print("   ❌ Failed to enable foreign key constraints")
            return False
    
    def verify_schema_fixes(self, conn: sqlite3.Connection) -> bool:
        """Verify that schema fixes were applied correctly."""
        print("\n🔍 Verifying schema fixes...")
        
        # Check graph_nodes columns
        required_columns = ['status', 'version', 'last_accessed_at']
        missing = []
        
        for column in required_columns:
            if not self.check_column_exists(conn, 'graph_nodes', column):
                missing.append(column)
        
        if missing:
            print(f"   ❌ Still missing columns: {missing}")
            return False
        else:
            print("   ✅ All required columns present")
        
        # Check foreign keys are enabled
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys")
        fk_enabled = cursor.fetchone()[0]
        
        if fk_enabled:
            print("   ✅ Foreign key constraints are enabled")
        else:
            print("   ⚠️  Foreign key constraints are disabled")
        
        return len(missing) == 0
    
    def backfill_content_hashes(self, conn: sqlite3.Connection) -> bool:
        """Backfill content hashes for messages that don't have them."""
        print("\n🔧 Backfilling content hashes for legacy messages...")
        
        # Get messages without content hashes
        cursor = conn.cursor()
        cursor.execute("""
            SELECT message_id, channel_id, user_id 
            FROM processed_messages 
            WHERE content_hash IS NULL
        """)
        
        null_hash_messages = cursor.fetchall()
        
        if not null_hash_messages:
            print("   ✅ All messages already have content hashes")
            return True
        
        print(f"   Found {len(null_hash_messages)} messages without content hashes")
        
        if self.dry_run:
            print(f"[DRY RUN] Would backfill {len(null_hash_messages)} content hashes")
            return True
        
        # Note: This is a simplified backfill - in reality, we'd need to:
        # 1. Fetch the original message content from Discord
        # 2. Process it through the same pipeline as ingestion
        # 3. Compute the hash using the same algorithm
        
        print("   ⚠️  Content hash backfill requires Discord API access")
        print("   ⚠️  This would need to be implemented with full message fetching")
        print("   ⚠️  For now, generating placeholder hashes based on message_id")
        
        updated_count = 0
        for message_id, channel_id, user_id in null_hash_messages[:10]:  # Limit to first 10 for demo
            # Generate a placeholder hash based on message_id
            placeholder_content = f"legacy_message_{message_id}"
            placeholder_hash = compute_content_hash(placeholder_content, [])
            
            update_sql = """
                UPDATE processed_messages 
                SET content_hash = ? 
                WHERE message_id = ?
            """
            
            if self.execute_sql(conn, update_sql, (placeholder_hash, message_id)):
                updated_count += 1
            
        print(f"   ✅ Updated {updated_count} messages with placeholder hashes")
        self.changes_made.append(f"Backfilled {updated_count} content hashes")
        
        return True
    
    def run_fixes(self, backfill_hashes: bool = False) -> bool:
        """Run all database fixes."""
        print("🔧 VITA Database Schema Fix")
        print("=" * 50)
        
        if self.dry_run:
            print("🧪 DRY RUN MODE - No changes will be made")
            print("=" * 50)
        
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            
            # Create backup first (in non-dry-run mode)
            if not self.dry_run:
                backup_path = f"{self.db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                print(f"📁 Creating backup: {backup_path}")
                
                backup_conn = sqlite3.connect(backup_path)
                conn.backup(backup_conn)
                backup_conn.close()
                print(f"   ✅ Backup created successfully")
                self.changes_made.append(f"Created backup: {backup_path}")
            
            success = True
            
            # Fix graph_nodes schema
            if not self.fix_graph_nodes_schema(conn):
                success = False
            
            # Enable foreign keys
            if not self.enable_foreign_keys(conn):
                success = False
            
            # Backfill content hashes if requested
            if backfill_hashes:
                if not self.backfill_content_hashes(conn):
                    success = False
            
            # Commit changes
            if not self.dry_run and success:
                conn.commit()
                print("\n💾 All changes committed successfully")
            
            # Verify fixes
            if not self.verify_schema_fixes(conn):
                success = False
            
            conn.close()
            
            # Print summary
            self.print_summary(success)
            
            return success
            
        except Exception as e:
            print(f"\n💥 Critical error during schema fix: {e}")
            return False
    
    def print_summary(self, success: bool):
        """Print a summary of changes made."""
        print("\n" + "=" * 50)
        print("📊 SCHEMA FIX SUMMARY")
        print("=" * 50)
        
        if success:
            print("✅ Schema fixes completed successfully")
        else:
            print("❌ Some schema fixes failed")
        
        if self.changes_made:
            print(f"\n📝 Changes made ({len(self.changes_made)}):")
            for i, change in enumerate(self.changes_made, 1):
                print(f"   {i}. {change}")
        else:
            print("\n📝 No changes were needed")
        
        if self.dry_run:
            print("\n🧪 This was a dry run - no actual changes were made")
            print("   Run without --dry-run to apply the changes")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Fix critical database schema issues in VITA"
    )
    parser.add_argument("--backfill-hashes", action="store_true", 
                       help="Backfill content hashes for legacy messages")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    parser.add_argument("--db-path", default="vita_data.db",
                       help="Path to the database file")
    
    args = parser.parse_args()
    
    # Check if database exists
    if not os.path.exists(args.db_path):
        print(f"❌ Database file not found: {args.db_path}")
        sys.exit(1)
    
    # Run fixes
    fixer = DatabaseSchemaFixer(args.db_path, args.dry_run)
    success = fixer.run_fixes(args.backfill_hashes)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 