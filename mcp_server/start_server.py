"""
Startup script for Adobe Analytics MCP Server
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.local")

if __name__ == "__main__":
    # Get configuration from environment
    host = os.environ.get("MCP_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_SERVER_PORT", "8001"))
    
    print(f"Starting Adobe Analytics MCP Server on {host}:{port}")
    
    uvicorn.run(
        "adobe_analytics_server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )