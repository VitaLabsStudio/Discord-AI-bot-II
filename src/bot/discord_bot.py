import os
import aiohttp
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import discord
from discord.ext import commands
from dotenv import load_dotenv
from ..backend.logger import get_logger
from ..backend.utils import extract_message_url

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

class VitaDiscordBot(commands.Bot):
    """VITA Discord AI Knowledge Assistant Bot."""
    
    def __init__(self):
        # Set up intents
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True
        intents.guilds = True
        
        super().__init__(
            command_prefix="!vita ",
            intents=intents,
            description="VITA AI Knowledge Assistant"
        )
        
        # Configuration
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        self.api_key = os.getenv("BACKEND_API_KEY")
        
        # HTTP session for backend requests
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def setup_hook(self):
        """Setup hook called when bot is ready."""
        # Create persistent HTTP session
        self.session = aiohttp.ClientSession(
            headers={"X-API-Key": self.api_key},
            timeout=aiohttp.ClientTimeout(total=120)  # Increased to 2 minutes for heavy processing
        )
        
        # Sync commands
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} command(s)")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")
    
    async def close(self):
        """Clean up resources when bot shuts down."""
        if self.session:
            await self.session.close()
        await super().close()
    
    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f"Bot is ready! Logged in as {self.user}")
        logger.info(f"Connected to {len(self.guilds)} guilds")
    
    async def on_message(self, message: discord.Message):
        """Handle incoming messages for ingestion."""
        # Skip bot messages
        if message.author.bot:
            return

        # Determine the message context for logging
        if isinstance(message.channel, discord.Thread):
            source_description = f"thread '{message.channel.name}' ({message.channel.id})"
        else:
            source_description = f"channel '#{message.channel.name}' ({message.channel.id})"

        logger.info(f"--- Message Received ---")
        logger.info(f"Message ID: {message.id} from User: {message.author.id}")
        logger.info(f"Source: {source_description}")
        logger.info(f"Content length: {len(message.content)}, Attachments: {len(message.attachments)}")

        # Skip if no content and no attachments
        if not message.content and not message.attachments:
            logger.info("Skipping message: No content or attachments.")
            return
        
        try:
            # Extract user roles
            user_roles = []
            if hasattr(message.author, 'roles'):
                user_roles = [role.name for role in message.author.roles]
            
            # Prepare attachment URLs
            attachment_urls = []
            if message.attachments:
                attachment_urls = [attachment.url for attachment in message.attachments]
            
            # Create ingestion request - FIXED: Properly handle thread vs channel IDs
            if isinstance(message.channel, discord.Thread):
                # Message is in a thread
                channel_id = str(message.channel.parent.id)  # Parent channel ID
                thread_id = str(message.channel.id)          # Thread ID
            else:
                # Message is in a regular channel
                channel_id = str(message.channel.id)
                thread_id = None
            
            ingest_data = {
                "message_id": str(message.id),
                "channel_id": channel_id,
                "user_id": str(message.author.id),
                "content": message.content or "",
                "timestamp": message.created_at.isoformat(),
                "attachments": attachment_urls if attachment_urls else None,
                "thread_id": thread_id,
                "roles": user_roles if user_roles else None
            }
            
            # Log before sending to backend
            logger.info(f"Sending message {message.id} to backend for ingestion.")
            
            # Send to backend for processing
            await self._send_to_backend("/ingest", ingest_data)
            
        except Exception as e:
            logger.error(f"Failed to process message {message.id}: {e}")
    
    async def _send_to_backend(self, endpoint: str, data: Dict[str, Any] = None, max_retries: int = 2, method: str = "POST") -> Optional[Dict]:
        """
        Send data to backend API with retry logic.
        
        Args:
            endpoint: API endpoint
            data: Data to send (for POST requests)
            max_retries: Maximum number of retry attempts
            method: HTTP method ("POST" or "GET")
            
        Returns:
            Response data or None if failed
        """
        if not self.session:
            logger.error("HTTP session not initialized")
            return None
        
        url = f"{self.backend_url}{endpoint}"
        
        for attempt in range(max_retries + 1):
            try:
                if method.upper() == "GET":
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 500 and attempt < max_retries:
                            logger.warning(f"Backend server error (attempt {attempt + 1}/{max_retries + 1}), retrying...")
                            await asyncio.sleep(2 ** attempt)
                            continue
                        else:
                            logger.error(f"Backend GET request failed: {response.status} - {await response.text()}")
                            return None
                else:  # POST
                    async with self.session.post(url, json=data) as response:
                        if response.status in [200, 202]:
                            return await response.json()
                        elif response.status == 500 and attempt < max_retries:
                            # Retry on server errors
                            logger.warning(f"Backend server error (attempt {attempt + 1}/{max_retries + 1}), retrying...")
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        else:
                            logger.error(f"Backend request failed: {response.status} - {await response.text()}")
                            return None
                        
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries + 1}), retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    logger.error(f"Request timeout after {max_retries + 1} attempts")
                    return None
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}, retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    logger.error(f"Failed to send request to backend after {max_retries + 1} attempts: {e}")
                    return None
        
        return None

# Slash Commands
@discord.app_commands.describe(
    question="Your question for the AI knowledge assistant"
)
async def ask_command(interaction: discord.Interaction, question: str):
    """Ask a question to the AI knowledge assistant."""
    try:
        await interaction.response.defer()
        
        # Get user roles
        user_roles = []
        if hasattr(interaction.user, 'roles'):
            user_roles = [role.name for role in interaction.user.roles]
        
        # Prepare query request
        query_data = {
            "user_id": str(interaction.user.id),
            "channel_id": str(interaction.channel.id),
            "roles": user_roles,
            "question": question,
            "top_k": 5
        }
        
        # Send query to backend
        bot = interaction.client
        response = await bot._send_to_backend("/query", query_data)
        
        if not response:
            await interaction.followup.send("❌ Sorry, I couldn't process your question right now. Please try again later.")
            return
        
        # Format response
        answer = response.get("answer", "No answer generated.")
        confidence = response.get("confidence", 0.0)
        citations = response.get("citations", [])
        
        # Create embed
        embed = discord.Embed(
            title="🤖 VITA AI Assistant",
            description=answer,
            color=0x00ff00 if confidence > 0.7 else 0xffaa00 if confidence > 0.4 else 0xff0000
        )
        
        # Add confidence indicator
        confidence_text = f"{confidence:.1%}"
        if confidence > 0.7:
            confidence_text += " 🟢"
        elif confidence > 0.4:
            confidence_text += " 🟡"
        else:
            confidence_text += " 🔴"
        
        embed.add_field(name="Confidence", value=confidence_text, inline=True)
        embed.add_field(name="Sources", value=str(len(citations)), inline=True)
        
        # Add citations if available
        if citations:
            citation_text = ""
            for i, citation in enumerate(citations[:3], 1):  # Limit to 3 citations
                message_id = citation.get("message_id", "")
                channel_id = citation.get("channel_id", "")
                guild_id = str(interaction.guild.id) if interaction.guild else ""
                
                if message_id and channel_id and guild_id:
                    message_url = extract_message_url(guild_id, channel_id, message_id)
                    citation_text += f"{i}. [Message]({message_url})\n"
                else:
                    citation_text += f"{i}. Message {message_id[:8]}...\n"
            
            if citation_text:
                embed.add_field(name="📚 Sources", value=citation_text, inline=False)
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Failed to process ask command: {e}")
        await interaction.followup.send("❌ An error occurred while processing your question. Please try again later.")

@discord.app_commands.describe(
    limit="Number of messages per channel to ingest (default: 100, max: 1000). Threads will be fully processed."
)
async def ingest_history_command(interaction: discord.Interaction, limit: int = 100):
    """Ingest message history from ALL channels and threads with real-time progress tracking."""
    try:
        # Check permissions - require Administrator for server-wide ingestion
        if not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message("❌ You need 'Administrator' permission to ingest server-wide history.", ephemeral=True)
            return
        
        # Validate limit
        if limit > 1000:
            limit = 1000
        elif limit < 1:
            limit = 1
        
        await interaction.response.defer()
        
        # Get all text channels in the guild
        guild = interaction.guild
        if not guild:
            await interaction.followup.send("❌ This command can only be used in a server.")
            return
        
        text_channels = [channel for channel in guild.text_channels if channel.permissions_for(guild.me).read_messages]
        
        if not text_channels:
            await interaction.followup.send("❌ No accessible text channels found.")
            return
        
        # Initial status message with enhanced design
        embed = discord.Embed(
            title="🚀 VITA Knowledge Ingestion Started",
            description="Processing server-wide message history with real-time tracking...",
            color=0x00aaff
        )
        embed.add_field(name="📊 Overall Progress", value="Initializing...", inline=False)
        embed.add_field(name="📁 Current Channel", value="Starting...", inline=True)
        embed.add_field(name="📈 Live Status", value="Preparing...", inline=True)
        embed.add_field(name="🔄 Recent Activity", value="Getting ready...", inline=False)
        embed.set_footer(text="This message updates in real-time • Use enhanced logging system")
        
        status_message = await interaction.followup.send(embed=embed)
        
        # Tracking variables
        total_messages_submitted = 0
        total_successful = 0
        total_failed = 0
        processed_channels = 0
        skipped_channels = 0
        current_batch_id = None
        
        bot = interaction.client
        
        for channel_idx, channel in enumerate(text_channels):
            try:
                # Check permissions
                if not channel.permissions_for(guild.me).read_message_history:
                    skipped_channels += 1
                    continue
                
                # Update embed for current channel
                progress_text = f"Channel {channel_idx + 1}/{len(text_channels)} • Processed: {processed_channels}"
                embed.set_field_at(0, name="📊 Overall Progress", value=progress_text, inline=False)
                embed.set_field_at(1, name="📁 Current Channel", value=f"#{channel.name}", inline=True)
                embed.set_field_at(2, name="📈 Live Status", value="🔍 Collecting messages...", inline=True)
                await status_message.edit(embed=embed)
                
                # Collect messages from this channel
                channel_messages = []
                async for message in channel.history(limit=limit):
                    if message.author.bot or (not message.content and not message.attachments):
                        continue
                    
                    # Extract user roles
                    user_roles = []
                    if hasattr(message.author, 'roles'):
                        user_roles = [role.name for role in message.author.roles]
                    
                    # Prepare attachment URLs
                    attachment_urls = []
                    if message.attachments:
                        attachment_urls = [attachment.url for attachment in message.attachments]
                    
                    # Create message data
                    message_data = {
                        "message_id": str(message.id),
                        "channel_id": str(channel.id),
                        "user_id": str(message.author.id),
                        "content": message.content or "",
                        "timestamp": message.created_at.isoformat(),
                        "attachments": attachment_urls if attachment_urls else None,
                        "thread_id": None,
                        "roles": user_roles if user_roles else None
                    }
                    
                    channel_messages.append(message_data)
                
                # Process channel messages with tracking
                if channel_messages:
                    # Update status
                    embed.set_field_at(2, name="📈 Live Status", value=f"🚀 Processing {len(channel_messages)} messages...", inline=True)
                    await status_message.edit(embed=embed)
                    
                    # Send batch to backend
                    batch_data = {"messages": channel_messages}
                    response = await bot._send_to_backend("/batch_ingest", batch_data)
                    
                    if response and response.get("batch_id"):
                        current_batch_id = response["batch_id"]
                        total_messages_submitted += len(channel_messages)
                        
                        # Monitor progress in real-time
                        await monitor_batch_progress(
                            bot, current_batch_id, status_message, embed, 
                            channel.name, channel_idx + 1, len(text_channels)
                        )
                        
                        # Get final counts from the last progress update
                        final_progress = await bot._send_to_backend(f"/progress/{current_batch_id}", {}, method="GET")
                        if final_progress:
                            progress_data = final_progress.get("progress", {})
                            total_successful += progress_data.get("success_count", 0)
                            total_failed += progress_data.get("error_count", 0)
                    else:
                        logger.error(f"Failed to start batch processing for #{channel.name}")
                        total_failed += len(channel_messages)
                
                # Process threads in this channel with similar tracking
                logger.info(f"Processing threads in channel: #{channel.name}")
                embed.set_field_at(2, name="📈 Live Status", value="🧵 Processing threads...", inline=True)
                await status_message.edit(embed=embed)
                
                try:
                    # Get all threads
                    all_threads = list(channel.threads)
                    async for thread in channel.archived_threads(limit=None):
                        all_threads.append(thread)
                    
                    # Process each thread
                    for thread in all_threads:
                        try:
                            if not thread.permissions_for(guild.me).read_message_history:
                                continue
                            
                            # Collect thread messages
                            thread_messages = []
                            async for message in thread.history(limit=None):
                                if message.author.bot or (not message.content and not message.attachments):
                                    continue
                                
                                user_roles = []
                                if hasattr(message.author, 'roles'):
                                    user_roles = [role.name for role in message.author.roles]
                                
                                attachment_urls = []
                                if message.attachments:
                                    attachment_urls = [attachment.url for attachment in message.attachments]
                                
                                message_data = {
                                    "message_id": str(message.id),
                                    "channel_id": str(channel.id),
                                    "user_id": str(message.author.id),
                                    "content": message.content or "",
                                    "timestamp": message.created_at.isoformat(),
                                    "attachments": attachment_urls if attachment_urls else None,
                                    "thread_id": str(thread.id),
                                    "roles": user_roles if user_roles else None
                                }
                                
                                thread_messages.append(message_data)
                            
                            # Process thread messages
                            if thread_messages:
                                embed.set_field_at(2, name="📈 Live Status", value=f"🧵 Thread: {thread.name[:20]}...", inline=True)
                                await status_message.edit(embed=embed)
                                
                                batch_data = {"messages": thread_messages}
                                response = await bot._send_to_backend("/batch_ingest", batch_data)
                                
                                if response and response.get("batch_id"):
                                    current_batch_id = response["batch_id"]
                                    total_messages_submitted += len(thread_messages)
                                    
                                    await monitor_batch_progress(
                                        bot, current_batch_id, status_message, embed,
                                        f"Thread: {thread.name}", channel_idx + 1, len(text_channels)
                                    )
                                    
                                    # Update counts
                                    final_progress = await bot._send_to_backend(f"/progress/{current_batch_id}", {}, method="GET")
                                    if final_progress:
                                        progress_data = final_progress.get("progress", {})
                                        total_successful += progress_data.get("success_count", 0)
                                        total_failed += progress_data.get("error_count", 0)
                                
                                await asyncio.sleep(0.2)  # Small delay between threads
                            
                        except Exception as e:
                            logger.error(f"Error processing thread '{thread.name}': {e}")
                            continue
                    
                except Exception as e:
                    logger.error(f"Error fetching threads for #{channel.name}: {e}")
                
                processed_channels += 1
                await asyncio.sleep(0.5)  # Small delay between channels
                
            except Exception as e:
                logger.error(f"Error processing channel #{channel.name}: {e}")
                skipped_channels += 1
                continue
        
        # Final status update with enhanced summary
        success_rate = (total_successful/total_messages_submitted*100) if total_messages_submitted > 0 else 0
        final_embed = discord.Embed(
            title="✅ VITA Knowledge Ingestion Complete!",
            description=f"Successfully processed **{processed_channels}** channels with **{success_rate:.1f}% success rate**",
            color=0x00ff00
        )
        
        # Primary stats
        final_embed.add_field(name="📊 Messages Submitted", value=f"**{total_messages_submitted:,}**", inline=True)
        final_embed.add_field(name="✅ Successfully Stored", value=f"**{total_successful:,}**", inline=True)
        final_embed.add_field(name="❌ Failed/Errors", value=f"**{total_failed:,}**", inline=True)
        
        # Processing stats
        final_embed.add_field(name="📁 Channels", value=f"✅ {processed_channels} | ⚠️ {skipped_channels}", inline=True)
        final_embed.add_field(name="📈 Success Rate", value=f"**{success_rate:.1f}%**", inline=True)
        final_embed.add_field(name="🚀 Enhanced Tracking", value="**Enabled**", inline=True)
        
        if skipped_channels > 0:
            final_embed.add_field(
                name="ℹ️ Note", 
                value=f"{skipped_channels} channels were skipped due to permission restrictions.",
                inline=False
            )
        
        final_embed.set_footer(text="Knowledge base successfully updated with enhanced tracking")
        
        try:
            await status_message.edit(embed=final_embed)
        except:
            await interaction.followup.send(embed=final_embed)
        
    except Exception as e:
        logger.error(f"Failed to process ingest_history command: {e}")
        await interaction.followup.send("❌ An error occurred during the ingestion process. Please check the logs for details.")

async def monitor_batch_progress(bot, batch_id: str, status_message, embed, 
                               current_location: str, channel_num: int, total_channels: int):
    """
    Monitor batch progress and update Discord message with real-time logs.
    """
    max_updates = 30  # Maximum number of progress updates
    update_count = 0
    
    while update_count < max_updates:
        try:
            # Get progress from backend
            progress_response = await bot._send_to_backend(f"/progress/{batch_id}", {}, method="GET")
            
            if not progress_response:
                await asyncio.sleep(2)
                update_count += 1
                continue
            
            progress = progress_response.get("progress", {})
            status = progress.get("status", "UNKNOWN")
            
            # Update progress display
            processed = progress.get("processed_count", 0)
            total = progress.get("total_messages", 0)
            success = progress.get("success_count", 0)
            errors = progress.get("error_count", 0)
            
            # Update embed fields with clearer progress display
            progress_bar = create_progress_bar(processed, total)
            channel_progress = f"Channel {channel_num}/{total_channels}"
            batch_progress = f"Batch: {progress_bar} ({processed}/{total})"
            overall_progress = f"{channel_progress}\n{batch_progress}"
            embed.set_field_at(0, name="📊 Overall Progress", value=overall_progress, inline=False)
            embed.set_field_at(1, name="📁 Current Location", value=current_location, inline=True)
            
            status_emoji = "🔄" if status == "PROCESSING" else "✅" if status == "COMPLETED" else "❌"
            status_text = f"{status_emoji} {status}\n✅ {success} | ❌ {errors}"
            embed.set_field_at(2, name="📈 Live Status", value=status_text, inline=True)
            
            # Show recent activity logs with better formatting
            recent_logs = progress.get("recent_logs", [])
            if recent_logs:
                latest_log = recent_logs[-1]
                log_status = latest_log.get("status", "UNKNOWN")
                log_emoji = "✅" if log_status == "SUCCESS" else "❌" if log_status == "ERROR" else "⏭️"
                
                # Get the most relevant details from the log
                details = latest_log.get("details", [])
                message_preview = latest_log.get('message_id', 'Unknown')[-6:]
                recent_activity = f"{log_emoji} Msg {message_preview}: {log_status}"
                
                if details:
                    # Show most relevant details (last 2, but prioritize important ones)
                    relevant_details = []
                    for detail in details:
                        if any(keyword in detail.lower() for keyword in ['stored', 'generated', 'processed', 'chunks', 'error']):
                            relevant_details.append(detail)
                    
                    # Use relevant details if found, otherwise use last 2
                    display_details = relevant_details[-2:] if relevant_details else details[-2:]
                    
                    for detail in display_details:
                        if len(detail) > 45:
                            detail = detail[:42] + "..."
                        recent_activity += f"\n• {detail}"
                
                embed.set_field_at(3, name="🔄 Recent Activity", value=f"```yaml\n{recent_activity[:180]}```", inline=False)
            
            # Update the message
            await status_message.edit(embed=embed)
            
            # Check if completed
            if status in ["COMPLETED", "FAILED"]:
                break
            
            await asyncio.sleep(2)  # Update every 2 seconds
            update_count += 1
            
        except Exception as e:
            logger.error(f"Error monitoring batch progress: {e}")
            await asyncio.sleep(2)
            update_count += 1

def create_progress_bar(current: int, total: int, length: int = 10) -> str:
    """Create a visual progress bar."""
    if total == 0:
        return "▱" * length
    
    filled = int((current / total) * length)
    bar = "▰" * filled + "▱" * (length - filled)
    percentage = (current / total) * 100
    return f"{bar} {percentage:.1f}%"

async def _send_batch_to_backend(bot: VitaDiscordBot, messages: List[Dict]) -> bool:
    """
    Helper function to send a batch of messages to the backend.
    
    Args:
        bot: The Discord bot instance
        messages: List of message data dictionaries
        
    Returns:
        True if successful, False otherwise
    """
    if not messages:
        return True
    
    try:
        batch_data = {"messages": messages}
        response = await bot._send_to_backend("/batch_ingest", batch_data)
        if response:
            logger.debug(f"Successfully sent batch of {len(messages)} messages to backend")
            return True
        else:
            logger.error(f"Failed to send batch of {len(messages)} messages to backend")
            return False
    except Exception as e:
        logger.error(f"Error sending batch to backend: {e}")
        return False

def setup_bot():
    """Create and configure the bot instance."""
    bot = VitaDiscordBot()
    
    # Add slash commands
    bot.tree.add_command(
        discord.app_commands.Command(
            name="ask",
            description="Ask a question to the AI knowledge assistant",
            callback=ask_command
        )
    )
    
    bot.tree.add_command(
        discord.app_commands.Command(
            name="ingest_history",
            description="Ingest channel message history and threads into the knowledge base",
            callback=ingest_history_command
        )
    )
    
    return bot

def run_discord_bot():
    """Run the Discord bot."""
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        logger.error("DISCORD_TOKEN environment variable not set")
        return
    
    bot = setup_bot()
    
    try:
        logger.info("Starting Discord bot...")
        bot.run(token, log_handler=None)  # Disable discord.py's default logging
    except Exception as e:
        logger.error(f"Failed to run Discord bot: {e}")
    finally:
        logger.info("Discord bot stopped") 