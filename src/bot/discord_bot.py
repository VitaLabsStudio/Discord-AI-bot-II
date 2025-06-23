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
            timeout=aiohttp.ClientTimeout(total=30)
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
        
        # Skip if no content and no attachments
        if not message.content and not message.attachments:
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
            
            # Send to backend for processing
            await self._send_to_backend("/ingest", ingest_data)
            
        except Exception as e:
            logger.error(f"Failed to process message {message.id}: {e}")
    
    async def _send_to_backend(self, endpoint: str, data: Dict[str, Any]) -> Optional[Dict]:
        """
        Send data to backend API.
        
        Args:
            endpoint: API endpoint
            data: Data to send
            
        Returns:
            Response data or None if failed
        """
        if not self.session:
            logger.error("HTTP session not initialized")
            return None
        
        try:
            url = f"{self.backend_url}{endpoint}"
            async with self.session.post(url, json=data) as response:
                if response.status in [200, 202]:
                    return await response.json()
                else:
                    logger.error(f"Backend request failed: {response.status} - {await response.text()}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to send request to backend: {e}")
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
    """Ingest message history from ALL channels and threads in the server into the knowledge base."""
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
        
        # Initial status message
        embed = discord.Embed(
            title="📚 Server-Wide History Ingestion Started",
            description=f"Processing {len(text_channels)} text channels and their threads...",
            color=0x00ff00
        )
        embed.add_field(name="Channels to Process", value=str(len(text_channels)), inline=True)
        embed.add_field(name="Messages per Channel", value=str(limit), inline=True)
        embed.add_field(name="Thread Processing", value="All messages (unlimited)", inline=True)
        embed.add_field(name="Status", value="🔄 Starting...", inline=False)
        
        status_message = await interaction.followup.send(embed=embed)
        
        # Process each channel
        total_messages = 0
        processed_channels = 0
        skipped_channels = 0
        total_threads = 0
        total_thread_messages = 0
        
        for channel in text_channels:
            try:
                # Check if bot has permission to read this channel
                if not channel.permissions_for(guild.me).read_message_history:
                    skipped_channels += 1
                    continue
                
                # Collect messages from this channel
                channel_messages = []
                async for message in channel.history(limit=limit):
                    # Skip bot messages
                    if message.author.bot:
                        continue
                    
                    # Skip if no content and no attachments
                    if not message.content and not message.attachments:
                        continue
                    
                    # Extract user roles
                    user_roles = []
                    if hasattr(message.author, 'roles'):
                        user_roles = [role.name for role in message.author.roles]
                    
                    # Prepare attachment URLs
                    attachment_urls = []
                    if message.attachments:
                        attachment_urls = [attachment.url for attachment in message.attachments]
                    
                    # Create message data - FIXED: Proper channel/thread ID handling
                    message_data = {
                        "message_id": str(message.id),
                        "channel_id": str(channel.id),  # This is the parent channel
                        "user_id": str(message.author.id),
                        "content": message.content or "",
                        "timestamp": message.created_at.isoformat(),
                        "attachments": attachment_urls if attachment_urls else None,
                        "thread_id": None,  # Regular channel messages have no thread_id
                        "roles": user_roles if user_roles else None
                    }
                    
                    channel_messages.append(message_data)
                
                # Send batch request for this channel if we have messages
                if channel_messages:
                    batch_data = {"messages": channel_messages}
                    bot = interaction.client
                    response = await bot._send_to_backend("/batch_ingest", batch_data)
                    
                    if response:
                        total_messages += len(channel_messages)
                        logger.info(f"Successfully ingested {len(channel_messages)} messages from #{channel.name}")
                    else:
                        logger.error(f"Failed to ingest messages from #{channel.name}")
                
                # Process threads in this channel
                logger.info(f"Processing threads in channel: #{channel.name}")
                
                try:
                    # Get all threads (both active and archived)
                    all_threads = []
                    
                    # Add active threads
                    all_threads.extend(channel.threads)
                    
                    # Add archived public threads
                    async for thread in channel.archived_threads(limit=None):
                        all_threads.append(thread)
                    
                    # Process each thread
                    for thread in all_threads:
                        try:
                            # Check if bot has permission to read this thread
                            if not thread.permissions_for(guild.me).read_message_history:
                                continue
                            
                            total_threads += 1
                            
                            thread_messages = []
                            async for message in thread.history(limit=None):
                                # Skip bot messages
                                if message.author.bot:
                                    continue
                                    
                                # Skip if no content and no attachments
                                if not message.content and not message.attachments:
                                    continue
                                
                                # Extract user roles
                                user_roles = []
                                if hasattr(message.author, 'roles'):
                                    user_roles = [role.name for role in message.author.roles]
                                
                                # Prepare attachment URLs
                                attachment_urls = []
                                if message.attachments:
                                    attachment_urls = [attachment.url for attachment in message.attachments]
                                
                                # Create message data - FIXED: Proper thread message handling
                                message_data = {
                                    "message_id": str(message.id),
                                    "channel_id": str(channel.id),  # Parent channel ID
                                    "user_id": str(message.author.id),
                                    "content": message.content or "",
                                    "timestamp": message.created_at.isoformat(),
                                    "attachments": attachment_urls if attachment_urls else None,
                                    "thread_id": str(thread.id),  # Properly set thread_id for thread messages
                                    "roles": user_roles if user_roles else None
                                }
                                
                                thread_messages.append(message_data)
                            
                            # Send batch request for this thread if we have messages
                            if thread_messages:
                                batch_data = {"messages": thread_messages}
                                response = await bot._send_to_backend("/batch_ingest", batch_data)
                                
                                if response:
                                    total_messages += len(thread_messages)
                                    total_thread_messages += len(thread_messages)
                                    logger.info(f"Successfully ingested {len(thread_messages)} messages from thread '{thread.name}' in #{channel.name}")
                                else:
                                    logger.error(f"Failed to ingest messages from thread '{thread.name}' in #{channel.name}")
                            
                            # Small delay between threads to prevent rate limiting
                            await asyncio.sleep(0.2)
                            
                        except Exception as e:
                            logger.error(f"Error processing thread '{thread.name}' in #{channel.name}: {e}")
                            continue
                    
                    if all_threads:
                        logger.info(f"Completed processing {len(all_threads)} threads in #{channel.name}")
                    
                except Exception as e:
                    logger.error(f"Error fetching threads for #{channel.name}: {e}")
                
                processed_channels += 1
                
                # Update progress every 5 channels
                if processed_channels % 5 == 0:
                    progress_embed = discord.Embed(
                        title="📚 Server-Wide History Ingestion In Progress",
                        description=f"Processing channels and threads... ({processed_channels}/{len(text_channels)})",
                        color=0xffaa00
                    )
                    progress_embed.add_field(name="Processed Channels", value=f"{processed_channels}/{len(text_channels)}", inline=True)
                    progress_embed.add_field(name="Total Messages", value=str(total_messages), inline=True)
                    progress_embed.add_field(name="Threads Processed", value=str(total_threads), inline=True)
                    progress_embed.add_field(name="Current Channel", value=f"#{channel.name}", inline=True)
                    
                    try:
                        await status_message.edit(embed=progress_embed)
                    except:
                        pass  # Ignore edit failures
                
                # Small delay to prevent rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing channel #{channel.name}: {e}")
                skipped_channels += 1
                continue
        
        # Final status update
        final_embed = discord.Embed(
            title="✅ Server-Wide History Ingestion Complete",
            description="Successfully processed all accessible channels and threads!",
            color=0x00ff00
        )
        final_embed.add_field(name="📊 Total Messages Ingested", value=str(total_messages), inline=True)
        final_embed.add_field(name="📁 Channels Processed", value=str(processed_channels), inline=True)
        final_embed.add_field(name="🧵 Threads Processed", value=str(total_threads), inline=True)
        final_embed.add_field(name="💬 Thread Messages", value=str(total_thread_messages), inline=True)
        final_embed.add_field(name="⚠️ Channels Skipped", value=str(skipped_channels), inline=True)
        final_embed.add_field(name="🏆 Status", value="Complete", inline=False)
        
        if skipped_channels > 0:
            final_embed.add_field(
                name="ℹ️ Note", 
                value=f"{skipped_channels} channels were skipped due to permission restrictions.",
                inline=False
            )
        
        try:
            await status_message.edit(embed=final_embed)
        except:
            # If edit fails, send new message
            await interaction.followup.send(embed=final_embed)
        
    except Exception as e:
        logger.error(f"Failed to process ingest_history command: {e}")
        await interaction.followup.send("❌ An error occurred while processing the server-wide history ingestion. Please try again later.")

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