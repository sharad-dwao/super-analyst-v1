# AI SDK Python Streaming with MCP Integration

This template demonstrates the usage of [Data Stream Protocol](https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol#data-stream-protocol) to stream chat completions from a Python endpoint ([FastAPI](https://fastapi.tiangolo.com)) with **MCP (Model Context Protocol)** integration for modular analytics operations.

## Architecture Overview

### MCP Integration Benefits
- **Modular Design**: Analytics operations are handled by separate MCP servers
- **Independent Deployment**: MCP servers can be updated without affecting the main application
- **Scalability**: Multiple MCP servers can handle different analytics functions
- **Fault Tolerance**: Robust error handling and fallback mechanisms

### System Components

1. **Main Application** (`api/index.py`)
   - FastAPI server handling chat requests
   - Streams responses using Data Stream Protocol
   - Communicates with MCP servers for analytics operations

2. **MCP Pipeline** (`api/utils/mcp_pipeline.py`)
   - Orchestrates analytics queries through MCP servers
   - Handles query enhancement and data analysis
   - Provides streaming support for real-time feedback

3. **MCP Client** (`api/utils/mcp_client.py`)
   - Generic MCP client for server communication
   - Handles tool discovery and execution
   - Provides error handling and retry logic

4. **Adobe Analytics MCP Server** (`mcp_server/adobe_analytics_server.py`)
   - Standalone server for Adobe Analytics operations
   - Implements MCP protocol for tool discovery and execution
   - Can be deployed and updated independently

## Deploy your own

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fvercel-labs%2Fai-sdk-preview-python-streaming&env=OPENROUTER_API_KEY,ADOBE_CLIENT_ID,ADOBE_CLIENT_SECRET,ADOBE_COMPANY_ID,ADOBE_ORG_ID,ADOBE_REPORTSUIT_ID,MCP_SERVER_URL&envDescription=API%20keys%20and%20configuration%20needed%20for%20application&envLink=https%3A%2F%2Fgithub.com%2Fvercel-labs%2Fai-sdk-preview-python-streaming%2Fblob%2Fmain%2F.env.example)

## How to use

### Prerequisites

1. Sign up for accounts with the AI providers you want to use (e.g., OpenAI, Anthropic).
2. Obtain API keys for each provider.
3. Set up Adobe Analytics API access and obtain required credentials.
4. Set the required environment variables as shown in the `.env.example` file.

### Installation

```bash
# Clone the repository
npx create-next-app --example https://github.com/vercel-labs/ai-sdk-preview-python-streaming ai-sdk-mcp-example

cd ai-sdk-mcp-example

# Install Node.js dependencies
pnpm install

# Create Python virtual environment
virtualenv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install MCP server dependencies
cd mcp_server
pip install -r requirements.txt
cd ..

# Copy environment variables
cp .env.example .env.local
# Edit .env.local with your actual API keys and configuration
```

### Development

#### Option 1: Run everything together (recommended for development)
```bash
pnpm dev
```
This starts:
- Next.js frontend on http://localhost:3000
- FastAPI main server on http://localhost:8000
- Adobe Analytics MCP server on http://localhost:8001

#### Option 2: Run without MCP server (for testing main app only)
```bash
pnpm run dev-no-mcp
```

#### Option 3: Run components separately
```bash
# Terminal 1: Next.js frontend
pnpm run next-dev

# Terminal 2: Main FastAPI server
pnpm run fastapi-dev

# Terminal 3: MCP server
pnpm run mcp-server-dev
```

### MCP Server Management

#### Check MCP Server Status
```bash
curl http://localhost:8001/health
```

#### List Available Tools
```bash
curl http://localhost:8001/tools
```

#### Deploy MCP Server Separately
The MCP server can be deployed independently:

```bash
cd mcp_server
python start_server.py
```

Or using Docker:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY mcp_server/requirements.txt .
RUN pip install -r requirements.txt
COPY mcp_server/ .
CMD ["python", "start_server.py"]
```

### Environment Variables

Create a `.env.local` file with the following variables:

```env
# OpenRouter API Key for LLM
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Adobe Analytics Configuration
ADOBE_CLIENT_ID=your_adobe_client_id
ADOBE_CLIENT_SECRET=your_adobe_client_secret
ADOBE_COMPANY_ID=your_adobe_company_id
ADOBE_ORG_ID=your_adobe_org_id
ADOBE_REPORTSUIT_ID=your_adobe_reportsuit_id

# MCP Server Configuration
MCP_SERVER_URL=http://localhost:8001
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8001
```

## MCP Integration Details

### Available MCP Tools

1. **get_analytics_report**: Get Adobe Analytics report for specified metrics and dimensions
2. **get_comparison_report**: Get comparison report between two time periods
3. **validate_schema**: Validate metrics and dimensions against Adobe Analytics schema
4. **get_current_date**: Get current server date

### MCP Communication Flow

1. **Tool Discovery**: Main application discovers available tools from MCP server
2. **Query Enhancement**: LLM enhances user queries with proper analytics context
3. **Tool Execution**: MCP client calls appropriate tools on MCP server
4. **Data Analysis**: LLM analyzes results and provides insights
5. **Streaming Response**: Results are streamed back to the user interface

### Benefits of MCP Architecture

- **Modularity**: Each analytics function is a separate, testable component
- **Scalability**: Multiple MCP servers can handle different data sources
- **Maintainability**: Updates to analytics logic don't require main app deployment
- **Reliability**: Fault-tolerant communication with proper error handling
- **Flexibility**: Easy to add new analytics tools or data sources

## API Endpoints

### Main Application
- `POST /api/chat` - Handle chat requests with streaming responses
- `GET /api/mcp/status` - Check MCP server status

### MCP Server
- `POST /mcp` - Handle MCP protocol requests
- `GET /health` - Health check endpoint
- `GET /tools` - List available tools

## Learn More

To learn more about the technologies used:

- [AI SDK Documentation](https://sdk.vercel.ai/docs)
- [Next.js Documentation](https://nextjs.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Adobe Analytics API](https://developer.adobe.com/analytics-apis/docs/2.0/)