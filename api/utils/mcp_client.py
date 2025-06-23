"""
MCP Client for Adobe Analytics
SECURITY: Only communicates with internal MCP servers
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
import httpx
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class MCPRequest(BaseModel):
    method: str = Field(description="MCP method name")
    params: Dict[str, Any] = Field(description="Method parameters")
    id: Optional[str] = Field(default=None, description="Request ID")


class MCPResponse(BaseModel):
    result: Optional[Dict[str, Any]] = Field(default=None, description="Response result")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error information")
    id: Optional[str] = Field(default=None, description="Request ID")


class MCPTool(BaseModel):
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    input_schema: Dict[str, Any] = Field(description="JSON schema for tool input")


class MCPClient:
    """Client for communicating with MCP servers"""
    
    def __init__(self, server_url: str, timeout: int = 30):
        if not self._is_internal_url(server_url):
            raise ValueError(f"MCP server URL must be internal: {server_url}")
        
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.session = None
        
    def _is_internal_url(self, url: str) -> bool:
        """Validate that URL is internal/localhost only"""
        internal_hosts = ["localhost", "127.0.0.1", "::1"]
        return any(host in url for host in internal_hosts)
        
    async def __aenter__(self):
        self.session = httpx.AsyncClient(
            timeout=self.timeout,
            verify=False
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def list_tools(self) -> List[MCPTool]:
        """List available tools from MCP server"""
        try:
            request = MCPRequest(
                method="tools/list",
                params={},
                id="list_tools"
            )
            
            response = await self._send_request(request)
            
            if response.error:
                logger.error(f"Error listing tools: {response.error}")
                return []
            
            tools = []
            for tool_data in response.result.get("tools", []):
                try:
                    tools.append(MCPTool(**tool_data))
                except Exception as e:
                    logger.warning(f"Failed to parse tool data: {tool_data}, error: {e}")
            
            logger.info(f"Found {len(tools)} tools from MCP server")
            return tools
            
        except Exception as e:
            logger.error(f"Failed to list tools: {str(e)}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool on the MCP server"""
        try:
            request = MCPRequest(
                method="tools/call",
                params={
                    "name": tool_name,
                    "arguments": arguments
                },
                id=f"call_{tool_name}"
            )
            
            logger.info(f"Calling MCP tool: {tool_name} with args: {arguments}")
            response = await self._send_request(request)
            
            if response.error:
                logger.error(f"Error calling tool {tool_name}: {response.error}")
                return {
                    "error": response.error,
                    "success": False
                }
            
            content = response.result.get("content", [])
            if content and len(content) > 0:
                try:
                    result_text = content[0].get("text", "{}")
                    result_data = json.loads(result_text)
                    logger.info(f"Tool {tool_name} executed successfully")
                    return result_data
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse tool result as JSON: {e}")
                    return {
                        "result": content,
                        "success": True
                    }
            else:
                logger.warning(f"No content in tool response for {tool_name}")
                return {
                    "result": response.result,
                    "success": True
                }
            
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name}: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }
    
    async def _send_request(self, request: MCPRequest) -> MCPResponse:
        """Send request to MCP server with retry logic"""
        if not self.session:
            raise RuntimeError("MCP client session not initialized")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.session.post(
                    f"{self.server_url}/mcp",
                    json=request.dict(),
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                response_data = response.json()
                return MCPResponse(**response_data)
                
            except httpx.HTTPError as e:
                logger.error(f"HTTP error communicating with MCP server (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error communicating with MCP server: {str(e)}")
                raise


class AdobeAnalyticsMCPClient:
    """Specialized MCP client for Adobe Analytics operations"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.client = MCPClient(server_url)
    
    async def get_analytics_report(
        self,
        metrics: List[str],
        dimensions: List[str],
        start_date: str,
        end_date: str,
        limit: int = 20
    ) -> Dict[str, Any]:
        """Get analytics report via MCP server"""
        async with self.client as mcp:
            return await mcp.call_tool(
                "get_analytics_report",
                {
                    "metrics": metrics,
                    "dimensions": dimensions,
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": limit
                }
            )
    
    async def get_comparison_report(
        self,
        metrics: List[str],
        dimensions: List[str],
        primary_start: str,
        primary_end: str,
        comparison_start: str,
        comparison_end: str,
        limit: int = 20
    ) -> Dict[str, Any]:
        """Get comparison report via MCP server"""
        async with self.client as mcp:
            return await mcp.call_tool(
                "get_comparison_report",
                {
                    "metrics": metrics,
                    "dimensions": dimensions,
                    "primary_start": primary_start,
                    "primary_end": primary_end,
                    "comparison_start": comparison_start,
                    "comparison_end": comparison_end,
                    "limit": limit
                }
            )
    
    async def validate_schema(
        self,
        metrics: List[str],
        dimensions: List[str]
    ) -> Dict[str, Any]:
        """Validate metrics and dimensions via MCP server"""
        async with self.client as mcp:
            return await mcp.call_tool(
                "validate_schema",
                {
                    "metrics": metrics,
                    "dimensions": dimensions
                }
            )
    
    async def get_current_date(self) -> Dict[str, Any]:
        """Get current date via MCP server"""
        async with self.client as mcp:
            return await mcp.call_tool("get_current_date", {})
    
    async def list_available_tools(self) -> List[MCPTool]:
        """List all available analytics tools"""
        async with self.client as mcp:
            return await mcp.list_tools()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check MCP server health"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.server_url}/health")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"MCP server health check failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }