import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import date, datetime, timedelta
import calendar
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from dotenv import load_dotenv

load_dotenv(".env.local")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Adobe Analytics MCP Server", 
    version="1.0.0", 
    docs_url=None, 
    redoc_url=None
)

ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Environment variables with validation
CLIENT_ID = os.environ.get("ADOBE_CLIENT_ID")
CLIENT_SECRET = os.environ.get("ADOBE_CLIENT_SECRET")
COMPANY_ID = os.environ.get("ADOBE_COMPANY_ID")
ORG_ID = os.environ.get("ADOBE_ORG_ID")
REPORTSUIT_ID = os.environ.get("ADOBE_REPORTSUIT_ID")

required_env_vars = {
    "ADOBE_CLIENT_ID": CLIENT_ID,
    "ADOBE_CLIENT_SECRET": CLIENT_SECRET,
    "ADOBE_COMPANY_ID": COMPANY_ID,
    "ADOBE_ORG_ID": ORG_ID,
    "ADOBE_REPORTSUIT_ID": REPORTSUIT_ID
}

missing_vars = [name for name, value in required_env_vars.items() if not value]
if missing_vars:
    logger.error(f"Missing required Adobe Analytics environment variables: {missing_vars}")
    raise ValueError(f"Missing required Adobe Analytics environment variables: {missing_vars}")

class MCPRequest(BaseModel):
    method: str = Field(description="MCP method name")
    params: Dict[str, Any] = Field(description="Method parameters")
    id: Optional[str] = Field(default=None, description="Request ID")

class MCPResponse(BaseModel):
    result: Optional[Dict[str, Any]] = Field(
        default=None, description="Response result"
    )
    error: Optional[Dict[str, Any]] = Field(
        default=None, description="Error information"
    )
    id: Optional[str] = Field(default=None, description="Request ID")

class MCPTool(BaseModel):
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    input_schema: Dict[str, Any] = Field(description="JSON schema for tool input")

# Comprehensive metrics and dimensions lists
METRICS = [
    "metrics/visits",
    "metrics/visitors",
    "metrics/pageviews",
    "metrics/bounces",
    "metrics/bouncerate",
    "metrics/entries",
    "metrics/exits",
    "metrics/orders",
    "metrics/revenue",
    "metrics/conversionrate",
    "metrics/averagetimespentonsite",
    "metrics/averagetimespentonpage",
    "metrics/averagevisitdepth",
    "metrics/units",
    "metrics/carts",
    "metrics/cartadditions",
    "metrics/cartremovals",
    "metrics/cartviews",
    "metrics/checkouts",
    "metrics/occurrences",
    "metrics/singlepagevisits",
    "metrics/reloads",
    "metrics/timespent",
    "metrics/campaigninstances",
    "metrics/clickthroughs",
]

DIMENSIONS = [
    "variables/page",
    "variables/pagename",
    "variables/pageurl",
    "variables/sitesection",
    "variables/referrer",
    "variables/referrertype",
    "variables/referringdomain",
    "variables/campaign",
    "variables/geocountry",
    "variables/georegion",
    "variables/geocity",
    "variables/browser",
    "variables/browsertype",
    "variables/operatingsystem",
    "variables/mobiledevicetype",
    "variables/mobiledevicename",
    "variables/marketingchannel",
    "variables/marketingchanneldetail",
    "variables/daterangemonth",
    "variables/daterangeweek",
    "variables/daterangeday",
    "variables/daterangeyear",
    "variables/daterangequarter",
    "variables/searchengine",
    "variables/searchenginekeyword",
    "variables/entrypage",
    "variables/exitpage",
    "variables/language",
    "variables/visitnumber",
]

# Token cache with thread safety
_token_cache = {"token": None, "expires_at": None}
_token_lock = asyncio.Lock()

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    client_host = request.client.host if request.client else "unknown"
    allowed_hosts = ["127.0.0.1", "localhost", "::1"]

    if client_host not in allowed_hosts:
        logger.warning(f"Blocked external access attempt from: {client_host}")
        raise HTTPException(
            status_code=403, detail="Access forbidden - internal use only"
        )

    response = await call_next(request)
    return response

async def get_access_token() -> str:
    async with _token_lock:
        current_time = datetime.now()

        if (
            _token_cache["token"]
            and _token_cache["expires_at"]
            and current_time < _token_cache["expires_at"]
        ):
            return _token_cache["token"]

        try:
            url = "https://ims-na1.adobelogin.com/ims/token/v3"
            payload = (
                f"grant_type=client_credentials&client_id={CLIENT_ID}"
                f"&client_secret={CLIENT_SECRET}"
                "&scope=openid%2CAdobeID%2Cadditional_info.projectedProductContext"
                "%2Ctarget_sdk%2Cread_organizations%2Cadditional_info.roles"
            )
            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(url, headers=headers, data=payload)
            resp.raise_for_status()

            token_data = resp.json()
            access_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)

            # Cache token with 5-minute buffer
            _token_cache["token"] = access_token
            _token_cache["expires_at"] = current_time + timedelta(seconds=expires_in - 300)

            logger.info("Adobe Analytics access token refreshed")
            return access_token

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting Adobe Analytics access token: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Failed to get Adobe Analytics access token: {str(e)}")
            raise

def validate_metrics_dimensions(
    metrics: List[str], dimensions: List[str]
) -> Dict[str, Any]:
    """Validate and clean metrics and dimensions"""
    valid_metrics = []
    invalid_metrics = []

    for metric in metrics:
        if metric in METRICS:
            valid_metrics.append(metric)
        else:
            # Try fuzzy matching
            metric_clean = metric.replace("metrics/", "").lower()
            for valid_metric in METRICS:
                if metric_clean in valid_metric.lower():
                    valid_metrics.append(valid_metric)
                    break
            else:
                invalid_metrics.append(metric)

    valid_dimensions = []
    invalid_dimensions = []

    for dimension in dimensions:
        if dimension in DIMENSIONS:
            valid_dimensions.append(dimension)
        else:
            # Try fuzzy matching
            dimension_clean = dimension.replace("variables/", "").lower()
            for valid_dimension in DIMENSIONS:
                if dimension_clean in valid_dimension.lower():
                    valid_dimensions.append(valid_dimension)
                    break
            else:
                invalid_dimensions.append(dimension)

    return {
        "valid_metrics": valid_metrics or ["metrics/visits"],
        "invalid_metrics": invalid_metrics,
        "valid_dimensions": valid_dimensions or ["variables/page"],
        "invalid_dimensions": invalid_dimensions,
    }

async def get_analytics_report(
    metrics: List[str],
    dimensions: List[str],
    start_date: str,
    end_date: str,
    limit: int = 20,
) -> Dict[str, Any]:
    """Get analytics report from Adobe Analytics API"""
    try:
        validation = validate_metrics_dimensions(metrics, dimensions)
        clean_metrics = validation["valid_metrics"]
        clean_dimensions = validation["valid_dimensions"]

        if validation["invalid_metrics"] or validation["invalid_dimensions"]:
            logger.warning(f"Invalid metrics/dimensions found: {validation}")

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json;charset=utf-8",
            "x-gw-ims-org-id": ORG_ID,
            "x-proxy-global-company-id": COMPANY_ID,
            "x-api-key": CLIENT_ID,
            "Authorization": f"Bearer {await get_access_token()}",
        }

        metric_entries = []
        for idx, metric in enumerate(clean_metrics):
            metric_entries.append({"columnId": str(idx), "id": metric})

        primary_dimension = clean_dimensions[0]

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
                "limit": min(limit, 50),
                "page": 0,
                "dimensionSort": "asc",
                "countRepeatInstances": True,
            },
        }

        # Add breakdown dimension if available
        if len(clean_dimensions) > 1:
            body["metricContainer"]["metricFilters"] = [
                {
                    "id": "0",
                    "type": "breakdown",
                    "dimension": clean_dimensions[1],
                    "itemId": "*",
                }
            ]

        url = f"https://analytics.adobe.io/api/{COMPANY_ID}/reports"

        logger.info(
            f"Adobe Analytics request: {len(clean_metrics)} metrics, {len(clean_dimensions)} dimensions"
        )
        
        async with httpx.AsyncClient(timeout=45) as client:
            res = await client.post(url, headers=headers, json=body)
        res.raise_for_status()

        response_data = res.json()

        return {
            "success": True,
            "data": response_data,
            "metadata": {
                "metrics": clean_metrics,
                "dimensions": clean_dimensions,
                "date_range": f"{start_date} to {end_date}",
                "total_rows": len(response_data.get("rows", [])),
                "server_type": "mcp_adobe_analytics",
                "validation": validation,
            },
        }

    except httpx.HTTPStatusError as err:
        error_details = {
            "success": False,
            "error": f"Adobe Analytics API error: {err.response.status_code} {err.response.reason_phrase}",
            "details": err.response.text if hasattr(err.response, "text") else str(err),
            "error_type": "api_error",
        }
        logger.error(f"Adobe Analytics API error: {error_details}")
        return error_details
    except Exception as e:
        error_details = {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "error_type": "internal_error",
        }
        logger.error(f"Unexpected error in get_analytics_report: {error_details}")
        return error_details

def parse_time_period(time_period: str, current_date: str) -> tuple[str, str]:
    """Parse time period string into start and end dates"""
    try:
        current = datetime.fromisoformat(current_date)
    except ValueError:
        logger.warning(f"Invalid current_date format: {current_date}, using today")
        current = datetime.now()

    # Handle specific month format (e.g., "november_2024")
    if "_" in time_period and any(
        month in time_period.lower()
        for month in [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ]
    ):
        try:
            month_name, year = time_period.lower().split("_")
            year = int(year)
            month_num = {
                "january": 1, "february": 2, "march": 3, "april": 4,
                "may": 5, "june": 6, "july": 7, "august": 8,
                "september": 9, "october": 10, "november": 11, "december": 12,
            }[month_name]

            first_day = datetime(year, month_num, 1)
            last_day = datetime(
                year, month_num, calendar.monthrange(year, month_num)[1]
            )

            return first_day.strftime("%Y-%m-%d"), last_day.strftime("%Y-%m-%d")
        except (ValueError, KeyError) as e:
            logger.warning(
                f"Could not parse specific month format: {time_period}, error: {e}"
            )

    # Handle relative time periods
    if "current_month" in time_period.lower() or "this month" in time_period.lower():
        first_day = current.replace(day=1)
        last_day = current.replace(
            day=calendar.monthrange(current.year, current.month)[1]
        )
        return first_day.strftime("%Y-%m-%d"), last_day.strftime("%Y-%m-%d")

    if "previous_month" in time_period.lower() or "last month" in time_period.lower():
        first_of_current = current.replace(day=1)
        last_of_previous = first_of_current - timedelta(days=1)
        first_of_previous = last_of_previous.replace(day=1)

        return first_of_previous.strftime("%Y-%m-%d"), last_of_previous.strftime(
            "%Y-%m-%d"
        )

    if "yesterday" in time_period.lower():
        date_obj = current - timedelta(days=1)
        return date_obj.strftime("%Y-%m-%d"), date_obj.strftime("%Y-%m-%d")
    elif "last week" in time_period.lower() or "past week" in time_period.lower():
        end_date = current - timedelta(days=1)
        start_date = end_date - timedelta(days=6)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    elif "last 30 days" in time_period.lower():
        end_date = current - timedelta(days=1)
        start_date = end_date - timedelta(days=29)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    elif "last 90 days" in time_period.lower():
        end_date = current - timedelta(days=1)
        start_date = end_date - timedelta(days=89)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    else:
        # Default to yesterday
        date_obj = current - timedelta(days=1)
        return date_obj.strftime("%Y-%m-%d"), date_obj.strftime("%Y-%m-%d")

# Define available MCP tools
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
                    "description": "List of metrics to retrieve (e.g., metrics/visits, metrics/pageviews)",
                },
                "dimensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of dimensions for breakdown (e.g., variables/page, variables/browser)",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Maximum number of rows to return (max 50)",
                },
            },
            "required": ["metrics", "dimensions", "start_date", "end_date"],
        },
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
                    "description": "List of metrics to retrieve",
                },
                "dimensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of dimensions for breakdown",
                },
                "primary_start": {
                    "type": "string",
                    "description": "Primary period start date (YYYY-MM-DD)",
                },
                "primary_end": {
                    "type": "string",
                    "description": "Primary period end date (YYYY-MM-DD)",
                },
                "comparison_start": {
                    "type": "string",
                    "description": "Comparison period start date (YYYY-MM-DD)",
                },
                "comparison_end": {
                    "type": "string",
                    "description": "Comparison period end date (YYYY-MM-DD)",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Maximum number of rows to return",
                },
            },
            "required": [
                "metrics",
                "dimensions",
                "primary_start",
                "primary_end",
                "comparison_start",
                "comparison_end",
            ],
        },
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
                    "description": "List of metrics to validate",
                },
                "dimensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of dimensions to validate",
                },
            },
            "required": ["metrics", "dimensions"],
        },
    ),
    MCPTool(
        name="get_current_date",
        description="Get current server date and time in ISO format",
        input_schema={"type": "object", "properties": {}, "required": []},
    ),
]

def ensure_json_serializable(obj: Any) -> Any:
    """Ensure object is JSON serializable"""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # Convert non-serializable objects to string
        return str(obj)

@app.post("/mcp")
async def handle_mcp_request(request: MCPRequest) -> MCPResponse:
    """Handle MCP protocol requests"""
    try:
        logger.info(f"MCP request: {request.method}")

        if request.method == "tools/list":
            return MCPResponse(
                result={"tools": [tool.dict() for tool in MCP_TOOLS]}, id=request.id
            )

        elif request.method == "tools/call":
            tool_name = request.params.get("name")
            arguments = request.params.get("arguments", {})

            logger.info(f"Calling tool: {tool_name}")

            if tool_name == "get_analytics_report":
                required_args = ["metrics", "dimensions", "start_date", "end_date"]
                missing_args = [arg for arg in required_args if arg not in arguments]
                if missing_args:
                    return MCPResponse(
                        error={
                            "code": -32602,
                            "message": f"Missing required arguments: {missing_args}",
                        },
                        id=request.id,
                    )

                result = await get_analytics_report(
                    metrics=arguments["metrics"],
                    dimensions=arguments["dimensions"],
                    start_date=arguments["start_date"],
                    end_date=arguments["end_date"],
                    limit=arguments.get("limit", 20),
                )

                # Ensure result is JSON serializable
                serializable_result = ensure_json_serializable(result)
                
                return MCPResponse(
                    result={
                        "content": [
                            {"type": "text", "text": json.dumps(serializable_result, indent=2, ensure_ascii=False)}
                        ]
                    },
                    id=request.id,
                )

            elif tool_name == "get_comparison_report":
                required_args = [
                    "metrics",
                    "dimensions",
                    "primary_start",
                    "primary_end",
                    "comparison_start",
                    "comparison_end",
                ]
                missing_args = [arg for arg in required_args if arg not in arguments]
                if missing_args:
                    return MCPResponse(
                        error={
                            "code": -32602,
                            "message": f"Missing required arguments: {missing_args}",
                        },
                        id=request.id,
                    )

                # Get both reports
                primary_result = await get_analytics_report(
                    metrics=arguments["metrics"],
                    dimensions=arguments["dimensions"],
                    start_date=arguments["primary_start"],
                    end_date=arguments["primary_end"],
                    limit=arguments.get("limit", 20),
                )

                comparison_result = await get_analytics_report(
                    metrics=arguments["metrics"],
                    dimensions=arguments["dimensions"],
                    start_date=arguments["comparison_start"],
                    end_date=arguments["comparison_end"],
                    limit=arguments.get("limit", 20),
                )

                result = {
                    "success": True,
                    "comparison_type": "period_comparison",
                    "primary_period": primary_result,
                    "comparison_period": comparison_result,
                    "metadata": {
                        "primary_dates": f"{arguments['primary_start']} to {arguments['primary_end']}",
                        "comparison_dates": f"{arguments['comparison_start']} to {arguments['comparison_end']}",
                        "server_type": "mcp_adobe_analytics",
                    },
                }

                # Ensure result is JSON serializable
                serializable_result = ensure_json_serializable(result)

                return MCPResponse(
                    result={
                        "content": [
                            {"type": "text", "text": json.dumps(serializable_result, indent=2, ensure_ascii=False)}
                        ]
                    },
                    id=request.id,
                )

            elif tool_name == "validate_schema":
                required_args = ["metrics", "dimensions"]
                missing_args = [arg for arg in required_args if arg not in arguments]
                if missing_args:
                    return MCPResponse(
                        error={
                            "code": -32602,
                            "message": f"Missing required arguments: {missing_args}",
                        },
                        id=request.id,
                    )

                validation = validate_metrics_dimensions(
                    arguments["metrics"], arguments["dimensions"]
                )

                result = {
                    "success": True,
                    "validation": validation,
                    "schema_info": {
                        "total_metrics_available": len(METRICS),
                        "total_dimensions_available": len(DIMENSIONS),
                        "server_type": "mcp_adobe_analytics",
                    },
                }

                # Ensure result is JSON serializable
                serializable_result = ensure_json_serializable(result)

                return MCPResponse(
                    result={
                        "content": [
                            {"type": "text", "text": json.dumps(serializable_result, indent=2, ensure_ascii=False)}
                        ]
                    },
                    id=request.id,
                )

            elif tool_name == "get_current_date":
                current_datetime = datetime.now()
                result = {
                    "success": True,
                    "date": current_datetime.strftime("%Y-%m-%d"),
                    "datetime": current_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    "iso_datetime": current_datetime.isoformat(),
                    "year": current_datetime.year,
                    "month": current_datetime.month,
                    "month_name": current_datetime.strftime("%B"),
                    "day": current_datetime.day,
                    "day_of_week": current_datetime.strftime("%A"),
                    "server_type": "mcp_adobe_analytics",
                }

                # For get_current_date, return just the date string as expected
                return MCPResponse(
                    result={"content": [{"type": "text", "text": result["date"]}]},
                    id=request.id,
                )

            else:
                return MCPResponse(
                    error={
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}. Available tools: {[tool.name for tool in MCP_TOOLS]}",
                    },
                    id=request.id,
                )

        else:
            return MCPResponse(
                error={
                    "code": -32601,
                    "message": f"Unknown method: {request.method}. Supported methods: tools/list, tools/call",
                },
                id=request.id,
            )

    except Exception as e:
        logger.error(f"Error handling MCP request: {str(e)}", exc_info=True)
        return MCPResponse(
            error={"code": -32603, "message": f"Internal error: {str(e)}"},
            id=request.id,
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        token = await get_access_token()
        adobe_status = "connected" if token else "disconnected"
    except Exception as e:
        adobe_status = f"error: {str(e)}"

    current_datetime = datetime.now()
    return {
        "status": "healthy",
        "server_type": "mcp_adobe_analytics",
        "version": "1.0.0",
        "tools_available": len(MCP_TOOLS),
        "adobe_analytics_status": adobe_status,
        "security": "internal_access_only",
        "current_date": current_datetime.strftime("%Y-%m-%d"),
        "current_datetime": current_datetime.strftime("%Y-%m-%d %H:%M:%S"),
    }

@app.get("/tools")
async def list_tools():
    """List available tools endpoint"""
    return {
        "tools": [tool.dict() for tool in MCP_TOOLS],
        "server_type": "mcp_adobe_analytics",
        "total_tools": len(MCP_TOOLS),
    }

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Starting Adobe Analytics MCP Server")

    try:
        token = await get_access_token()
        logger.info("Adobe Analytics connection validated successfully")
    except Exception as e:
        logger.error(f"Failed to validate Adobe Analytics connection: {str(e)}")

    logger.info(f"MCP Server started with {len(MCP_TOOLS)} tools available")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)