# VITA Discord AI Knowledge Assistant

An advanced Discord bot powered by AI that can ingest, process, and answer questions about your server's message history. VITA uses OpenAI's GPT models with Retrieval-Augmented Generation (RAG) to provide intelligent responses based on your Discord server's knowledge base.

## 🌟 Features

### Core Functionality
- **🤖 AI-Powered Q&A**: Ask questions and get intelligent answers based on your server's message history
- **📚 Server-Wide Ingestion**: Automatically processes ALL channels and threads in your Discord server
- **🧵 Thread Support**: Full support for Discord threads including real-time ingestion and historical processing
- **📄 Document Processing**: Supports DOCX, PDF, images (OCR), and other file formats
- **🔒 Permission-Aware**: Respects Discord permissions and roles for secure knowledge access
- **⚡ Real-Time Processing**: New messages are automatically ingested as they're posted

### Advanced Features
- **🎯 Semantic Search**: Uses vector embeddings for intelligent content retrieval
- **📊 Confidence Scoring**: Provides confidence levels for AI responses
- **🔗 Source Citations**: Links back to original Discord messages
- **🚀 Async Processing**: High-performance asynchronous architecture
- **📈 Progress Tracking**: Real-time progress updates during bulk ingestion
- **🛡️ Rate Limiting**: Built-in protections against API rate limits

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Discord Bot Token
- OpenAI API Key
- Pinecone API Key

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/VitaLabsStudio/Discord-AI-bot-II.git
   cd Discord-AI-bot-II
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp env.example .env
   ```
   
   Edit the `.env` file with your actual configuration:
   ```env
   DISCORD_TOKEN=your_discord_bot_token_here
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   BACKEND_API_KEY=your_secure_backend_api_key_here
   ```

5. **Run the application**:
   ```bash
   python src/main.py
   ```

## 🔧 Configuration

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DISCORD_TOKEN` | Your Discord bot token | `MTEx...` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `PINECONE_API_KEY` | Pinecone vector database key | `12345...` |
| `BACKEND_API_KEY` | Internal API authentication key | `secure_random_string` |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PINECONE_INDEX_NAME` | `vita-knowledge-base` | Pinecone index name |
| `OPENAI_CHAT_MODEL` | `gpt-4o` | OpenAI chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `BACKEND_HOST` | `0.0.0.0` | Backend server host |
| `BACKEND_PORT` | `8000` | Backend server port |
| `LOG_LEVEL` | `INFO` | Logging level |

## 🎮 Discord Setup

### 1. Create Discord Application
1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application" and give it a name
3. Go to "Bot" section and click "Add Bot"
4. Copy the bot token to your `.env` file
5. Enable "Message Content Intent" under "Privileged Gateway Intents"

### 2. Invite Bot to Server
1. Go to "OAuth2" > "URL Generator"
2. Select scopes: `bot`, `applications.commands`
3. Select bot permissions:
   - Read Messages/View Channels
   - Send Messages
   - Use Slash Commands
   - Read Message History
   - Embed Links
4. Use the generated URL to invite the bot to your server

### 3. Set Up API Keys

#### OpenAI API
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key to your `.env` file

#### Pinecone API
1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Create a new project and index
3. Copy the API key to your `.env` file

## 💬 Usage

### Slash Commands

#### `/ask <question>`
Ask a question to the AI knowledge assistant.
```
/ask What was discussed about the new feature last week?
```

#### `/ingest_history [limit]`
Ingest message history from all channels and threads in the server.
- `limit`: Number of messages per channel (default: 100, max: 1000)
- Threads are always fully processed regardless of limit
- Requires Administrator permission

```
/ingest_history 500
```

### Automatic Features
- **Real-time ingestion**: New messages are automatically processed
- **Thread support**: Messages in threads are automatically captured
- **Document processing**: File attachments are automatically processed

## 🏗️ Architecture

```
VITA Discord AI Bot
├── src/
│   ├── main.py                 # Application entry point
│   │   └── discord_bot.py      # Discord bot implementation
│   └── backend/
│       ├── api.py              # FastAPI backend
│       ├── ingestion.py        # Message processing pipeline
│       ├── embedding.py        # Vector embeddings management
│       ├── llm_client.py       # OpenAI integration
│       ├── file_processor.py   # Document processing
│       ├── permissions.py      # Permission management
│       ├── processed_log.py    # Idempotency tracking
│       ├── schemas.py          # Data models
│       ├── logger.py           # Logging configuration
│       └── utils.py            # Utility functions
├── requirements.txt            # Python dependencies
├── env.example                 # Environment template
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 🔒 Security & Privacy

- **API Key Protection**: All sensitive keys are stored in environment variables
- **Permission Respect**: Only processes messages the bot can legitimately access
- **Data Privacy**: No message content is logged or stored insecurely
- **Rate Limiting**: Built-in protections against API abuse
- **Secure Communication**: All API communications use secure headers

## 🛠️ Development

### Running in Development Mode
```bash
# Start backend only
python src/main.py --backend-only

# Start with debug logging
LOG_LEVEL=DEBUG python src/main.py
```

### Testing
```bash
# Run basic health checks
curl http://localhost:8000/health

# Test ingestion endpoint
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_backend_api_key" \
  -d '{"message_id": "test", "channel_id": "test", "user_id": "test", "content": "test message", "timestamp": "2023-01-01T00:00:00Z"}'
```

## 🚨 Troubleshooting

### Common Issues

1. **Bot not responding to commands**
   - Ensure bot has proper permissions in Discord
   - Check that "Message Content Intent" is enabled
   - Verify bot token is correct

2. **0 messages ingested**
   - Ensure bot has "Read Message History" permission
   - Check that channels aren't restricted
   - Verify API keys are configured correctly

3. **Document processing failures**
   - Install required system dependencies for file processing
   - Check OCR configuration if processing images

4. **API rate limiting**
   - The bot includes built-in rate limiting
   - Consider upgrading API plans for higher limits

### Logs
Check logs for detailed error information:
```bash
tail -f vita.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit with clear messages: `git commit -m "Add feature description"`
5. Push to your fork: `git push origin feature-name`
6. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing the GPT and embedding models
- Pinecone for the vector database platform
- Discord.py community for the excellent Discord library
- All contributors who helped improve this project

## 📞 Support

If you encounter any issues or have questions:

1. Check the [troubleshooting section](#troubleshooting)
2. Search existing [GitHub Issues](https://github.com/VitaLabsStudio/Discord-AI-bot-II/issues)
3. Create a new issue with detailed information
4. Join our Discord community for real-time support

---

**⚡ VITA - Bringing AI-powered knowledge management to your Discord community!** 