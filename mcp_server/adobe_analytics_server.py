"""
MCP Server for Adobe Analytics Operations
Standalone server that can be deployed and updated independently
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import date, datetime, timedelta
import calendar
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests
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

# Initialize FastAPI app for MCP server
app = FastAPI(title="Adobe Analytics MCP Server", version="1.0.0")

# Adobe Analytics configuration
CLIENT_ID = os.environ.get("ADOBE_CLIENT_ID")
CLIENT_SECRET = os.environ.get("ADOBE_CLIENT_SECRET")
COMPANY_ID = os.environ.get("ADOBE_COMPANY_ID")
ORG_ID = os.environ.get("ADOBE_ORG_ID")
REPORTSUIT_ID = os.environ.get("ADOBE_REPORTSUIT_ID")


class MCPRequest(BaseModel):
    """MCP request structure"""
    method: str = Field(description="MCP method name")
    params: Dict[str, Any] = Field(description="Method parameters")
    id: Optional[str] = Field(default=None, description="Request ID")


class MCPResponse(BaseModel):
    """MCP response structure"""
    result: Optional[Dict[str, Any]] = Field(default=None, description="Response result")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error information")
    id: Optional[str] = Field(default=None, description="Request ID")


class MCPTool(BaseModel):
    """MCP tool definition"""
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    input_schema: Dict[str, Any] = Field(description="JSON schema for tool input")


# Adobe Analytics schema (simplified for MCP server)
METRICS = [
    "metrics/visits", "metrics/visitors", "metrics/pageviews", "metrics/bounces",
    "metrics/bouncerate", "metrics/entries", "metrics/exits", "metrics/orders",
    "metrics/revenue", "metrics/conversionrate", "metrics/averagetimespentonsite"
]

DIMENSIONS = [
    "variables/page", "variables/pagename", "variables/referrer", "variables/campaign",
    "variables/geocountry", "variables/browser", "variables/mobiledevicetype",
    "variables/marketingchannel", "variables/daterangemonth", "variables/daterangeweek",
    "variables/daterangeday"
]


def get_access_token() -> str:
    """Get Adobe Analytics access token"""
    url = "https://ims-na1.adobelogin.com/ims/token/v3"
    payload = (
        f"grant_type=client_credentials&client_id={CLIENT_ID}"
        f"&client_secret={CLIENT_SECRET}"
        "&scope=openid%2CAdobeID%2Cadditional_info.projectedProductContext"
        "%2Ctarget_sdk%2Cread_organizations%2Cadditional_info.roles"
    )
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    resp = requests.post(url, headers=headers, data=payload, timeout=3)
    resp.raise_for_status()
    return resp.json()["access_token"]


def get_analytics_report(
    metrics: List[str],
    dimensions: List[str],
    start_date: str,
    end_date: str,
    limit: int = 20
) -> Dict[str, Any]:
    """Get Adobe Analytics report"""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json;charset=utf-8",
        "x-gw-ims-org-id": ORG_ID,
        "x-proxy-global-company-id": COMPANY_ID,
        "x-api-key": CLIENT_ID,
        "Authorization": f"Bearer {get_access_token()}",
    }

    # Prepare metrics
    metric_entries = []
    for idx, metric in enumerate(metrics):
        metric_entries.append({"columnId": str(idx), "id": metric})

    # Use primary dimension
    primary_dimension = dimensions[0] if dimensions else "variables/page"

    # Build request body
    body = {
        "rsid": REPORTSUIT_ID,
        "globalFilters": [
            {
                "type": "dateRange",
                "dateRange": f"{start_date}T00:00:00/{end_date}T23:59:59",
            }
        ],
        "metricContainer": {"metrics": metric_entries},
        "dimension": primary_dimension,
        "settings": {
            "limit": limit,
            "page": 0,
            "dimensionSort": "asc",
            "countRepeatInstances": True,
        },
    }

    # Add second dimension if provided
    if len(dimensions) > 1:
        body["metricContainer"]["metricFilters"] = [
            {
                "id": "0",
                "type": "breakdown",
                "dimension": dimensions[1],
                "itemId": "*",
            }
        ]

    url = f"https://analytics.adobe.io/api/{COMPANY_ID}/reports"

    try:
        logger.info(f"Making Adobe Analytics request: {json.dumps(body, indent=2)}")
        res = requests.post(url, headers=headers, json=body, timeout=30)
        res.raise_for_status()

        response_data = res.json()
        
        return {
            "success": True,
            "data": response_data,
            "metadata": {
                "metrics": metrics,
                "dimensions": dimensions,
                "date_range": f"{start_date} to {end_date}",
                "total_rows": len(response_data.get("rows", [])),
                "server_type": "mcp_adobe_analytics"
            }
        }

    except requests.exceptions.HTTPError as err:
        error_details = {
            "success": False,
            "error": f"{err.response.status_code} {err.response.reason}",
            "details": err.response.text,
            "request_body": body,
        }
        logger.error(f"Adobe Analytics API error: {error_details}")
        return error_details
    except Exception as e:
        error_details = {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "request_body": body
        }
        logger.error(f"Unexpected error: {error_details}")
        return error_details


def parse_time_period(time_period: str, current_date: str) -> tuple[str, str]:
    """Parse time period string into start and end dates"""
    current = datetime.fromisoformat(current_date)
    
    # Handle specific month formats
    if "_" in time_period and any(month in time_period.lower() for month in [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]):
        try:
            month_name, year = time_period.lower().split("_")
            year = int(year)
            month_num = {
                "january": 1, "february": 2, "march": 3, "april": 4,
                "may": 5, "june": 6, "july": 7, "august": 8,
                "september": 9, "october": 10, "november": 11, "december": 12
            }[month_name]
            
            first_day = datetime(year, month_num, 1)
            last_day = datetime(year, month_num, calendar.monthrange(year, month_num)[1])
            
            return first_day.strftime("%Y-%m-%d"), last_day.strftime("%Y-%m-%d")
        except (ValueError, KeyError):
            logger.warning(f"Could not parse specific month format: {time_period}")
    
    # Handle current/this month
    if "current_month" in time_period.lower() or "this month" in time_period.lower():
        first_day = current.replace(day=1)
        last_day = current.replace(day=calendar.monthrange(current.year, current.month)[1])
        return first_day.strftime("%Y-%m-%d"), last_day.strftime("%Y-%m-%d")
    
    # Handle previous/last month
    if "previous_month" in time_period.lower() or "last month" in time_period.lower():
        first_of_current = current.replace(day=1)
        last_of_previous = first_of_current - timedelta(days=1)
        first_of_previous = last_of_previous.replace(day=1)
        
        return first_of_previous.strftime("%Y-%m-%d"), last_of_previous.strftime("%Y-%m-%d")
    
    # Handle other time periods
    if "yesterday" in time_period.lower():
        date_obj = current - timedelta(days=1)
        return date_obj.strftime("%Y-%m-%d"), date_obj.strftime("%Y-%m-%d")
    elif "last week" in time_period.lower():
        end_date = current - timedelta(days=1)
        start_date = end_date - timedelta(days=6)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    elif "last 30 days" in time_period.lower():
        end_date = current - timedelta(days=1)
        start_date = end_date - timedelta(days=29)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    else:
        # Default to yesterday
        date_obj = current - timedelta(days=1)
        return date_obj.strftime("%Y-%m-%d"), date_obj.strftime("%Y-%m-%d")


# MCP Tool Definitions
MCP_TOOLS = [
    MCPTool(
        name="get_analytics_report",
        description="Get Adobe Analytics report for specified metrics and dimensions",
        input_schema={
            "type": "object",
            "properties": {
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of metrics to retrieve"
                },
                "dimensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of dimensions for breakdown"
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format"
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Maximum number of rows to return"
                }
            },
            "required": ["metrics", "dimensions", "start_date", "end_date"]
        }
    ),
    MCPTool(
        name="get_comparison_report",
        description="Get comparison report between two time periods",
        input_schema={
            "type": "object",
            "properties": {
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of metrics to retrieve"
                },
                "dimensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of dimensions for breakdown"
                },
                "primary_start": {"type": "string", "description": "Primary period start date"},
                "primary_end": {"type": "string", "description": "Primary period end date"},
                "comparison_start": {"type": "string", "description": "Comparison period start date"},
                "comparison_end": {"type": "string", "description": "Comparison period end date"},
                "limit": {"type": "integer", "default": 20}
            },
            "required": ["metrics", "dimensions", "primary_start", "primary_end", "comparison_start", "comparison_end"]
        }
    ),
    MCPTool(
        name="validate_schema",
        description="Validate metrics and dimensions against Adobe Analytics schema",
        input_schema={
            "type": "object",
            "properties": {
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of metrics to validate"
                },
                "dimensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of dimensions to validate"
                }
            },
            "required": ["metrics", "dimensions"]
        }
    ),
    MCPTool(
        name="get_current_date",
        description="Get current server date",
        input_schema={
            "type": "object",
            "properties": {},
            "required": []
        }
    )
]


@app.post("/mcp")
async def handle_mcp_request(request: MCPRequest) -> MCPResponse:
    """Handle MCP requests"""
    try:
        logger.info(f"Received MCP request: {request.method}")
        
        if request.method == "tools/list":
            return MCPResponse(
                result={
                    "tools": [tool.dict() for tool in MCP_TOOLS]
                },
                id=request.id
            )
        
        elif request.method == "tools/call":
            tool_name = request.params.get("name")
            arguments = request.params.get("arguments", {})
            
            if tool_name == "get_analytics_report":
                result = get_analytics_report(
                    metrics=arguments["metrics"],
                    dimensions=arguments["dimensions"],
                    start_date=arguments["start_date"],
                    end_date=arguments["end_date"],
                    limit=arguments.get("limit", 20)
                )
                
                return MCPResponse(
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2)
                            }
                        ]
                    },
                    id=request.id
                )
            
            elif tool_name == "get_comparison_report":
                # Get primary period data
                primary_result = get_analytics_report(
                    metrics=arguments["metrics"],
                    dimensions=arguments["dimensions"],
                    start_date=arguments["primary_start"],
                    end_date=arguments["primary_end"],
                    limit=arguments.get("limit", 20)
                )
                
                # Get comparison period data
                comparison_result = get_analytics_report(
                    metrics=arguments["metrics"],
                    dimensions=arguments["dimensions"],
                    start_date=arguments["comparison_start"],
                    end_date=arguments["comparison_end"],
                    limit=arguments.get("limit", 20)
                )
                
                result = {
                    "success": True,
                    "comparison_type": "period_comparison",
                    "primary_period": primary_result,
                    "comparison_period": comparison_result,
                    "metadata": {
                        "primary_dates": f"{arguments['primary_start']} to {arguments['primary_end']}",
                        "comparison_dates": f"{arguments['comparison_start']} to {arguments['comparison_end']}",
                        "server_type": "mcp_adobe_analytics"
                    }
                }
                
                return MCPResponse(
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2)
                            }
                        ]
                    },
                    id=request.id
                )
            
            elif tool_name == "validate_schema":
                valid_metrics = [m for m in arguments["metrics"] if m in METRICS]
                valid_dimensions = [d for d in arguments["dimensions"] if d in DIMENSIONS]
                
                result = {
                    "success": True,
                    "validation": {
                        "valid_metrics": valid_metrics,
                        "invalid_metrics": [m for m in arguments["metrics"] if m not in METRICS],
                        "valid_dimensions": valid_dimensions,
                        "invalid_dimensions": [d for d in arguments["dimensions"] if d not in DIMENSIONS]
                    },
                    "schema_info": {
                        "total_metrics_available": len(METRICS),
                        "total_dimensions_available": len(DIMENSIONS)
                    }
                }
                
                return MCPResponse(
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2)
                            }
                        ]
                    },
                    id=request.id
                )
            
            elif tool_name == "get_current_date":
                result = {
                    "success": True,
                    "date": date.today().isoformat(),
                    "server_type": "mcp_adobe_analytics"
                }
                
                return MCPResponse(
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": result["date"]
                            }
                        ]
                    },
                    id=request.id
                )
            
            else:
                return MCPResponse(
                    error={
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    },
                    id=request.id
                )
        
        else:
            return MCPResponse(
                error={
                    "code": -32601,
                    "message": f"Unknown method: {request.method}"
                },
                id=request.id
            )
    
    except Exception as e:
        logger.error(f"Error handling MCP request: {str(e)}")
        return MCPResponse(
            error={
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            },
            id=request.id
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "server_type": "mcp_adobe_analytics",
        "version": "1.0.0",
        "tools_available": len(MCP_TOOLS)
    }


@app.get("/tools")
async def list_tools():
    """List available tools"""
    return {
        "tools": [tool.dict() for tool in MCP_TOOLS],
        "server_type": "mcp_adobe_analytics"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)