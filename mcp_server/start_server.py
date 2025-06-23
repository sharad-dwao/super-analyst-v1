"""
Startup script for Adobe Analytics MCP Server
SECURITY: Configured for internal access only
"""

import uvicorn
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.local")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # SECURITY: Only bind to localhost/127.0.0.1 for internal access
    host = "127.0.0.1"  # Force localhost only
    port = int(os.environ.get("MCP_SERVER_PORT", "8001"))
    
    logger.info(f"Starting Adobe Analytics MCP Server on {host}:{port}")
    logger.info("SECURITY: Server configured for internal access only")
    
    # Validate required environment variables
    required_vars = [
        "ADOBE_CLIENT_ID", "ADOBE_CLIENT_SECRET", "ADOBE_COMPANY_ID", 
        "ADOBE_ORG_ID", "ADOBE_REPORTSUIT_ID"
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        exit(1)
    
    try:
        uvicorn.run(
            "adobe_analytics_server:app",
            host=host,
            port=port,
            reload=True,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"Failed to start MCP server: {str(e)}")
        exit(1)