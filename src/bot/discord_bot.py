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
    
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        """Handle reaction-based message deletion."""
        try:
            # Only handle 🛑 (stop sign) emoji
            if str(payload.emoji) != "🛑":
                return
            
            # Skip bot reactions
            if payload.user_id == self.user.id:
                return
            
            # Get the message and user
            channel = self.get_channel(payload.channel_id)
            if not channel:
                return
            
            message = await channel.fetch_message(payload.message_id)
            user = self.get_user(payload.user_id)
            
            # Check if the user reacting is the original message author or an admin
            is_author = str(message.author.id) == str(payload.user_id)
            is_admin = False
            
            if hasattr(user, 'guild_permissions') and user.guild_permissions:
                is_admin = user.guild_permissions.administrator
            elif payload.guild_id:
                guild = self.get_guild(payload.guild_id)
                if guild:
                    member = guild.get_member(payload.user_id)
                    if member and member.guild_permissions.administrator:
                        is_admin = True
            
            if not (is_author or is_admin):
                logger.info(f"User {payload.user_id} tried to delete message {payload.message_id} but lacks permission")
                return
            
            # Send delete request to backend
            response = await self._send_to_backend(f"/delete_message/{message.id}", method="DELETE")
            
            if response:
                # Add confirmation reaction
                await message.add_reaction("✅")
                logger.info(f"Message {message.id} vectors deleted by user {payload.user_id}")
            else:
                # Add error reaction
                await message.add_reaction("❌")
                logger.error(f"Failed to delete vectors for message {message.id}")
        
        except Exception as e:
            logger.error(f"Error handling reaction deletion: {e}")
    
    async def on_raw_reaction_remove(self, payload: discord.RawReactionActionEvent):
        """Handle reaction removal (cleanup only)."""
        # Currently no action needed for reaction removal
        pass
    
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
            method: HTTP method ("POST", "GET", or "DELETE")
            
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
                elif method.upper() == "DELETE":
                    async with self.session.delete(url) as response:
                        if response.status in [200, 202, 204]:
                            try:
                                return await response.json()
                            except:
                                # DELETE might not return JSON content
                                return {"status": "success"}
                        elif response.status == 500 and attempt < max_retries:
                            logger.warning(f"Backend server error (attempt {attempt + 1}/{max_retries + 1}), retrying...")
                            await asyncio.sleep(2 ** attempt)
                            continue
                        else:
                            logger.error(f"Backend DELETE request failed: {response.status} - {await response.text()}")
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

class FeedbackView(discord.ui.View):
    """UI View for collecting feedback on AI responses."""
    
    def __init__(self, bot: VitaDiscordBot, query_text: str, answer_text: str, user_id: str, confidence: float):
        super().__init__(timeout=300)  # 5 minute timeout
        self.bot = bot
        self.query_text = query_text
        self.answer_text = answer_text
        self.user_id = user_id
        self.confidence = confidence
        self.feedback_given = False
    
    @discord.ui.button(label="Helpful", emoji="👍", style=discord.ButtonStyle.green)
    async def helpful_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Handle helpful feedback."""
        if interaction.user.id != int(self.user_id):
            await interaction.response.send_message("❌ You can only give feedback on your own queries.", ephemeral=True)
            return
        
        if self.feedback_given:
            await interaction.response.send_message("✅ Feedback already recorded!", ephemeral=True)
            return
        
        await self._record_feedback(interaction, True)
    
    @discord.ui.button(label="Not Helpful", emoji="👎", style=discord.ButtonStyle.red)
    async def not_helpful_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Handle not helpful feedback."""
        if interaction.user.id != int(self.user_id):
            await interaction.response.send_message("❌ You can only give feedback on your own queries.", ephemeral=True)
            return
        
        if self.feedback_given:
            await interaction.response.send_message("✅ Feedback already recorded!", ephemeral=True)
            return
        
        await self._record_feedback(interaction, False)
    
    async def _record_feedback(self, interaction: discord.Interaction, is_helpful: bool):
        """Record feedback in the backend."""
        try:
            feedback_data = {
                "query_text": self.query_text,
                "answer_text": self.answer_text,
                "is_helpful": is_helpful,
                "user_id": self.user_id,
                "confidence_score": self.confidence
            }
            
            response = await self.bot._send_to_backend("/feedback", feedback_data)
            
            if response:
                # Disable all buttons
                for item in self.children:
                    item.disabled = True
                self.feedback_given = True
                
                # Update the original message to show feedback was recorded
                embed = interaction.message.embeds[0] if interaction.message.embeds else None
                if embed:
                    embed.add_field(
                        name="📝 Feedback",
                        value=f"Thank you! Your feedback has been recorded. {'👍' if is_helpful else '👎'}",
                        inline=False
                    )
                
                await interaction.response.edit_message(embed=embed, view=self)
                logger.info(f"Recorded {'helpful' if is_helpful else 'not helpful'} feedback from user {self.user_id}")
            else:
                await interaction.response.send_message("❌ Failed to record feedback. Please try again later.", ephemeral=True)
                
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            await interaction.response.send_message("❌ Error recording feedback. Please try again later.", ephemeral=True)
    
    async def on_timeout(self):
        """Handle view timeout."""
        for item in self.children:
            item.disabled = True
        # Note: We can't edit the message here without storing the interaction/message reference

# Slash Commands
@discord.app_commands.describe(
    days="Number of days to analyze (1-30, default: 7)"
)
async def digest_command(interaction: discord.Interaction, days: int = 7):
    """Generate a thematic digest of recent server activity."""
    await interaction.response.defer()
    
    try:
        # Validate days parameter
        if days < 1 or days > 30:
            await interaction.followup.send("❌ Days parameter must be between 1 and 30.")
            return
        
        # Call backend to generate digest
        data = {"days": days}
        bot = interaction.client
        response = await bot._send_to_backend("/digest", data)
        
        if not response or "digest" not in response:
            await interaction.followup.send("❌ Failed to generate digest. Please try again later.")
            return
        
        digest = response["digest"]
        
        # Create main embed
        embed = discord.Embed(
            title=f"📊 {digest.get('title', 'Thematic Digest')}",
            description=digest.get('summary', 'No summary available'),
            color=0x4CAF50,
            timestamp=datetime.utcnow()
        )
        
        # Add overview
        embed.add_field(
            name="📈 Overview",
            value=f"• **Documents Analyzed:** {digest.get('document_count', 0)}\n• **Key Themes:** {len(digest.get('themes', []))}\n• **Time Period:** Last {days} days",
            inline=False
        )
        
        # Add themes
        themes = digest.get('themes', [])
        if themes:
            for i, theme in enumerate(themes[:4], 1):  # Limit to 4 themes for Discord embed limits
                importance_emoji = "🔴" if theme.get('importance') == 'high' else "🟡" if theme.get('importance') == 'medium' else "🟢"
                theme_summary = theme.get('summary', 'No summary')
                if len(theme_summary) > 300:
                    theme_summary = theme_summary[:297] + "..."
                
                embed.add_field(
                    name=f"{importance_emoji} Theme {i}: {theme.get('theme_title', 'Unknown')}",
                    value=f"{theme_summary}\n\n*{theme.get('document_count', 0)} related messages*",
                    inline=False
                )
        else:
            embed.add_field(
                name="ℹ️ No Themes Identified",
                value="Not enough data for thematic analysis. Try with more days or check if there are recent conversations.",
                inline=False
            )
        
        embed.set_footer(text="VITA v4.0 • Thematic Analysis from server conversations")
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Failed to process digest command: {e}")
        await interaction.followup.send("❌ An error occurred while generating the digest. Please try again later.")

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
        
        # Add feedback instructions
        embed.add_field(
            name="💡 Help me improve!",
            value="Use the buttons below to rate this answer, or react with 🛑 to remove any of your messages from my knowledge base.",
            inline=False
        )
        
        # Create feedback view
        feedback_view = FeedbackView(
            bot=bot,
            query_text=question,
            answer_text=answer,
            user_id=str(interaction.user.id),
            confidence=confidence
        )
        
        await interaction.followup.send(embed=embed, view=feedback_view)
        
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

@discord.app_commands.describe()
async def dlq_stats_command(interaction: discord.Interaction):
    """View dead letter queue statistics (Admin only)."""
    try:
        # Check admin permissions
        if not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message("❌ You need 'Administrator' permission to view DLQ statistics.", ephemeral=True)
            return
        
        await interaction.response.defer()
        
        # Get DLQ stats from backend
        bot = interaction.client
        response = await bot._send_to_backend("/admin/dlq/stats", {}, method="GET")
        
        if not response:
            await interaction.followup.send("❌ Failed to fetch DLQ statistics.")
            return
        
        stats = response.get("stats", {})
        total_items = stats.get("total_items", 0)
        
        if total_items == 0:
            embed = discord.Embed(
                title="📊 Dead Letter Queue Statistics",
                description="✅ No failed items in the queue!",
                color=0x00ff00
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Create detailed stats embed
        embed = discord.Embed(
            title="📊 Dead Letter Queue Statistics",
            description=f"Found **{total_items}** failed items",
            color=0xff6600
        )
        
        # Failure type breakdown
        failure_types = stats.get("failure_types", {})
        if failure_types:
            failure_text = ""
            for failure_type, count in failure_types.items():
                emoji = {
                    "download": "⬇️", "parsing": "📄", "ocr": "👁️", 
                    "api": "🔌", "network": "🌐", "validation": "✅", "unknown": "❓"
                }.get(failure_type, "❓")
                failure_text += f"{emoji} {failure_type.title()}: **{count}**\n"
            embed.add_field(name="🔍 Failure Types", value=failure_text, inline=True)
        
        # Retry statistics
        retry_stats = stats.get("retry_stats", {})
        if retry_stats:
            retry_text = f"🆕 New: **{retry_stats.get('no_retries', 0)}**\n"
            retry_text += f"🔄 Retried: **{retry_stats.get('has_retries', 0)}**"
            embed.add_field(name="🔄 Retry Status", value=retry_text, inline=True)
        
        # Recent failures
        recent_24h = stats.get("recent_failures_24h", 0)
        recent_text = f"**{recent_24h}** in last 24h"
        embed.add_field(name="⏰ Recent Failures", value=recent_text, inline=True)
        
        # Time range
        if stats.get("oldest_failure") and stats.get("newest_failure"):
            from datetime import datetime
            oldest = datetime.fromisoformat(stats["oldest_failure"].replace('Z', '+00:00'))
            newest = datetime.fromisoformat(stats["newest_failure"].replace('Z', '+00:00'))
            
            time_text = f"📅 **Oldest:** {oldest.strftime('%Y-%m-%d %H:%M')}\n"
            time_text += f"📅 **Newest:** {newest.strftime('%Y-%m-%d %H:%M')}"
            embed.add_field(name="📈 Time Range", value=time_text, inline=False)
        
        embed.set_footer(text="Use /dlq_view to see specific failed items")
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Failed to process dlq_stats command: {e}")
        await interaction.followup.send("❌ An error occurred while fetching DLQ statistics.")

@discord.app_commands.describe(
    failure_type="Filter by failure type",
    limit="Number of items to show (max 20)"
)
async def dlq_view_command(interaction: discord.Interaction, failure_type: str = None, limit: int = 10):
    """View dead letter queue items (Admin only)."""
    try:
        # Check admin permissions
        if not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message("❌ You need 'Administrator' permission to view DLQ items.", ephemeral=True)
            return
        
        # Validate limit
        if limit > 20:
            limit = 20
        elif limit < 1:
            limit = 1
        
        await interaction.response.defer()
        
        # Prepare query parameters
        params = {"limit": limit}
        if failure_type:
            params["failure_type"] = failure_type
        
        # Get DLQ items from backend
        bot = interaction.client
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        response = await bot._send_to_backend(f"/admin/dlq/items?{query_string}", {}, method="GET")
        
        if not response:
            await interaction.followup.send("❌ Failed to fetch DLQ items.")
            return
        
        items = response.get("items", [])
        
        if not items:
            embed = discord.Embed(
                title="📋 Dead Letter Queue Items",
                description="✅ No failed items found!",
                color=0x00ff00
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Create items display
        embed = discord.Embed(
            title="📋 Dead Letter Queue Items",
            description=f"Showing **{len(items)}** failed items",
            color=0xff6600
        )
        
        for i, item in enumerate(items[:5]):  # Show max 5 items per embed
            failure_emoji = {
                "download": "⬇️", "parsing": "📄", "ocr": "👁️",
                "api": "🔌", "network": "🌐", "validation": "✅", "unknown": "❓"
            }.get(item.get("failure_type", "unknown"), "❓")
            
            # Format timestamp
            from datetime import datetime
            timestamp = datetime.fromisoformat(item["timestamp"].replace('Z', '+00:00'))
            time_str = timestamp.strftime('%m/%d %H:%M')
            
            # Create field value
            url_preview = item.get("url", "")[:50] + "..." if len(item.get("url", "")) > 50 else item.get("url", "")
            error_preview = item.get("error", "")[:100] + "..." if len(item.get("error", "")) > 100 else item.get("error", "")
            
            retry_count = item.get("retry_count", 0)
            retry_text = f" (🔄 {retry_count} retries)" if retry_count > 0 else ""
            
            field_value = f"**URL:** {url_preview}\n"
            field_value += f"**Error:** {error_preview}\n"
            field_value += f"**Step:** {item.get('step', 'unknown')}\n"
            field_value += f"**Time:** {time_str}{retry_text}"
            
            embed.add_field(
                name=f"{failure_emoji} {item.get('failure_type', 'unknown').title()} #{i+1}",
                value=field_value,
                inline=False
            )
        
        if len(items) > 5:
            embed.set_footer(text=f"Showing first 5 of {len(items)} items. Use limit parameter to see more.")
        else:
            embed.set_footer(text="Use /dlq_retry <item_id> to retry a specific item")
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Failed to process dlq_view command: {e}")
        await interaction.followup.send("❌ An error occurred while fetching DLQ items.")

@discord.app_commands.describe(
    days="Remove items older than this many days (default: 30)"
)
async def dlq_cleanup_command(interaction: discord.Interaction, days: int = 30):
    """Clean up old dead letter queue items (Admin only)."""
    try:
        # Check admin permissions
        if not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message("❌ You need 'Administrator' permission to cleanup DLQ.", ephemeral=True)
            return
        
        # Validate days
        if days < 1:
            days = 1
        elif days > 365:
            days = 365
        
        await interaction.response.defer()
        
        # Send cleanup request to backend
        bot = interaction.client
        response = await bot._send_to_backend(f"/admin/dlq/cleanup", {"days": days})
        
        if not response:
            await interaction.followup.send("❌ Failed to cleanup DLQ.")
            return
        
        removed_count = response.get("removed_count", 0)
        
        if removed_count == 0:
            embed = discord.Embed(
                title="🧹 DLQ Cleanup Complete",
                description="✅ No old items found to remove.",
                color=0x00ff00
            )
        else:
            embed = discord.Embed(
                title="🧹 DLQ Cleanup Complete",
                description=f"✅ Removed **{removed_count}** items older than {days} days.",
                color=0x00ff00
            )
        
        embed.set_footer(text="Cleanup helps maintain optimal DLQ performance")
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Failed to process dlq_cleanup command: {e}")
        await interaction.followup.send("❌ An error occurred during DLQ cleanup.")

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
            name="digest",
            description="Generate a thematic summary of recent server activity",
            callback=digest_command
        )
    )
    
    bot.tree.add_command(
        discord.app_commands.Command(
            name="ingest_history",
            description="Ingest channel message history and threads into the knowledge base",
            callback=ingest_history_command
        )
    )
    
    # Admin commands for DLQ management
    bot.tree.add_command(
        discord.app_commands.Command(
            name="dlq_stats",
            description="View dead letter queue statistics (Admin only)",
            callback=dlq_stats_command
        )
    )
    
    bot.tree.add_command(
        discord.app_commands.Command(
            name="dlq_view",
            description="View dead letter queue items (Admin only)",
            callback=dlq_view_command
        )
    )
    
    bot.tree.add_command(
        discord.app_commands.Command(
            name="dlq_cleanup",
            description="Clean up old dead letter queue items (Admin only)",
            callback=dlq_cleanup_command
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