import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
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
    result: Optional[Dict[str, Any]] = Field(default=None, description="Response result")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error information")
    id: Optional[str] = Field(default=None, description="Request ID")

class MCPTool(BaseModel):
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    input_schema: Dict[str, Any] = Field(description="JSON schema for tool input")

class DimensionFilter(BaseModel):
    dimension: str = Field(description="Dimension to filter on")
    operator: str = Field(description="Filter operator (equals, contains, starts_with, ends_with, not_equals)")
    values: List[str] = Field(description="Values to filter by")

class SegmentFilter(BaseModel):
    segment_id: str = Field(description="Adobe Analytics segment ID")

class GlobalFilter(BaseModel):
    type: str = Field(description="Filter type (dateRange, segment, dimension)")
    dateRange: Optional[str] = Field(default=None, description="Date range filter")
    segmentId: Optional[str] = Field(default=None, description="Segment ID")
    dimension: Optional[str] = Field(default=None, description="Dimension for filtering")
    itemId: Optional[str] = Field(default=None, description="Item ID for dimension filter")
    itemIds: Optional[List[str]] = Field(default=None, description="Multiple item IDs")

METRICS = [
    "metrics/visits", "metrics/visitors", "metrics/pageviews", "metrics/bounces",
    "metrics/bouncerate", "metrics/entries", "metrics/exits", "metrics/orders",
    "metrics/revenue", "metrics/conversionrate", "metrics/averagetimespentonsite",
    "metrics/averagetimespentonpage", "metrics/averagevisitdepth", "metrics/units",
    "metrics/carts", "metrics/cartadditions", "metrics/cartremovals", "metrics/cartviews",
    "metrics/checkouts", "metrics/occurrences", "metrics/singlepagevisits",
    "metrics/reloads", "metrics/timespent", "metrics/campaigninstances",
    "metrics/clickthroughs", "metrics/event1", "metrics/event2", "metrics/event3",
    "metrics/event4", "metrics/event5", "metrics/event6", "metrics/event7",
    "metrics/event8", "metrics/event9", "metrics/event10"
]

DIMENSIONS = [
    "variables/page", "variables/pagename", "variables/pageurl", "variables/sitesection",
    "variables/referrer", "variables/referrertype", "variables/referringdomain",
    "variables/campaign", "variables/geocountry", "variables/georegion",
    "variables/geocity", "variables/browser", "variables/browsertype",
    "variables/operatingsystem", "variables/mobiledevicetype", "variables/mobiledevicename",
    "variables/marketingchannel", "variables/marketingchanneldetail",
    "variables/daterangemonth", "variables/daterangeweek", "variables/daterangeday",
    "variables/daterangeyear", "variables/daterangequarter", "variables/searchengine",
    "variables/searchenginekeyword", "variables/entrypage", "variables/exitpage",
    "variables/language", "variables/visitnumber", "variables/prop1", "variables/prop2",
    "variables/prop3", "variables/prop4", "variables/prop5", "variables/evar1",
    "variables/evar2", "variables/evar3", "variables/evar4", "variables/evar5"
]

_token_cache = {"token": None, "expires_at": None}
_token_lock = asyncio.Lock()

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    client_host = request.client.host if request.client else "unknown"
    allowed_hosts = ["127.0.0.1", "localhost", "::1"]

    if client_host not in allowed_hosts:
        logger.warning(f"Blocked external access attempt from: {client_host}")
        raise HTTPException(status_code=403, detail="Access forbidden - internal use only")

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

def validate_metrics_dimensions(metrics: List[str], dimensions: List[str]) -> Dict[str, Any]:
    valid_metrics = []
    invalid_metrics = []

    for metric in metrics:
        if metric in METRICS:
            valid_metrics.append(metric)
        else:
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

def build_global_filters(
    start_date: str,
    end_date: str,
    segment_filters: Optional[List[SegmentFilter]] = None,
    dimension_filters: Optional[List[DimensionFilter]] = None
) -> List[Dict[str, Any]]:
    filters = []
    
    filters.append({
        "type": "dateRange",
        "dateRange": f"{start_date}T00:00:00/{end_date}T23:59:59"
    })
    
    if segment_filters:
        for segment_filter in segment_filters:
            filters.append({
                "type": "segment",
                "segmentId": segment_filter.segment_id
            })
    
    if dimension_filters:
        for dim_filter in dimension_filters:
            filter_obj = {
                "type": "dimension",
                "dimension": dim_filter.dimension
            }
            
            if dim_filter.operator == "equals":
                if len(dim_filter.values) == 1:
                    filter_obj["itemId"] = dim_filter.values[0]
                else:
                    filter_obj["itemIds"] = dim_filter.values
            elif dim_filter.operator == "contains":
                filter_obj["search"] = {
                    "clause": dim_filter.values[0],
                    "includeSearchTotal": True
                }
            elif dim_filter.operator == "starts_with":
                filter_obj["search"] = {
                    "clause": f"{dim_filter.values[0]}*",
                    "includeSearchTotal": True
                }
            elif dim_filter.operator == "ends_with":
                filter_obj["search"] = {
                    "clause": f"*{dim_filter.values[0]}",
                    "includeSearchTotal": True
                }
            elif dim_filter.operator == "not_equals":
                filter_obj["excludeItemIds"] = dim_filter.values
            
            filters.append(filter_obj)
    
    return filters

def build_breakdown_structure(dimensions: List[str]) -> Optional[Dict[str, Any]]:
    if len(dimensions) <= 1:
        return None
    
    breakdown = {
        "dimension": dimensions[1],
        "itemId": "*"
    }
    
    if len(dimensions) > 2:
        breakdown["breakdown"] = build_breakdown_structure(dimensions[1:])
    
    return breakdown

async def get_complex_analytics_report(
    metrics: List[str],
    dimensions: List[str],
    start_date: str,
    end_date: str,
    segment_filters: Optional[List[SegmentFilter]] = None,
    dimension_filters: Optional[List[DimensionFilter]] = None,
    limit: int = 20,
    sort_metric: Optional[str] = None,
    sort_direction: str = "desc"
) -> Dict[str, Any]:
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
            metric_entry = {"columnId": str(idx), "id": metric}
            if sort_metric and metric == sort_metric:
                metric_entry["sort"] = sort_direction
            metric_entries.append(metric_entry)

        global_filters = build_global_filters(
            start_date, end_date, segment_filters, dimension_filters
        )

        body = {
            "rsid": REPORTSUIT_ID,
            "globalFilters": global_filters,
            "metricContainer": {"metrics": metric_entries},
            "dimension": clean_dimensions[0],
            "settings": {
                "limit": min(limit, 400),
                "page": 0,
                "dimensionSort": "asc",
                "countRepeatInstances": True,
                "includeLatLong": False,
                "includeAnomalyDetection": False
            },
        }

        breakdown = build_breakdown_structure(clean_dimensions)
        if breakdown:
            body["metricContainer"]["metricFilters"] = [
                {
                    "id": "0",
                    "type": "breakdown",
                    **breakdown
                }
            ]

        url = f"https://analytics.adobe.io/api/{COMPANY_ID}/reports"

        logger.info(f"Complex Adobe Analytics request: {len(clean_metrics)} metrics, {len(clean_dimensions)} dimensions, {len(global_filters)} filters")
        
        async with httpx.AsyncClient(timeout=60) as client:
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
                "filters_applied": {
                    "segments": len(segment_filters) if segment_filters else 0,
                    "dimensions": len(dimension_filters) if dimension_filters else 0
                },
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
        logger.error(f"Unexpected error in get_complex_analytics_report: {error_details}")
        return error_details

async def get_analytics_report(
    metrics: List[str],
    dimensions: List[str],
    start_date: str,
    end_date: str,
    limit: int = 20,
) -> Dict[str, Any]:
    return await get_complex_analytics_report(
        metrics=metrics,
        dimensions=dimensions,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )

async def get_segments_list() -> Dict[str, Any]:
    try:
        headers = {
            "Accept": "application/json",
            "x-gw-ims-org-id": ORG_ID,
            "x-proxy-global-company-id": COMPANY_ID,
            "x-api-key": CLIENT_ID,
            "Authorization": f"Bearer {await get_access_token()}",
        }

        url = f"https://analytics.adobe.io/api/{COMPANY_ID}/segments"
        params = {
            "rsids": REPORTSUIT_ID,
            "limit": 100,
            "includeType": "all"
        }

        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.get(url, headers=headers, params=params)
        res.raise_for_status()

        response_data = res.json()

        return {
            "success": True,
            "segments": response_data.get("content", []),
            "total_segments": response_data.get("totalElements", 0),
            "server_type": "mcp_adobe_analytics"
        }

    except Exception as e:
        logger.error(f"Error getting segments list: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": "segments_error"
        }

async def get_dimension_values(
    dimension: str,
    search_term: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    try:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-gw-ims-org-id": ORG_ID,
            "x-proxy-global-company-id": COMPANY_ID,
            "x-api-key": CLIENT_ID,
            "Authorization": f"Bearer {await get_access_token()}",
        }

        body = {
            "rsid": REPORTSUIT_ID,
            "dimension": dimension,
            "settings": {
                "limit": min(limit, 50000),
                "page": 0
            }
        }

        if search_term:
            body["search"] = {
                "clause": search_term,
                "includeSearchTotal": True
            }

        url = f"https://analytics.adobe.io/api/{COMPANY_ID}/reports/dimensionValues"

        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(url, headers=headers, json=body)
        res.raise_for_status()

        response_data = res.json()

        return {
            "success": True,
            "dimension": dimension,
            "values": response_data.get("rows", []),
            "total_values": len(response_data.get("rows", [])),
            "search_term": search_term,
            "server_type": "mcp_adobe_analytics"
        }

    except Exception as e:
        logger.error(f"Error getting dimension values: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": "dimension_values_error"
        }

def parse_time_period(time_period: str, current_date: str) -> tuple[str, str]:
    try:
        current = datetime.fromisoformat(current_date)
    except ValueError:
        logger.warning(f"Invalid current_date format: {current_date}, using today")
        current = datetime.now()

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
            logger.warning(f"Could not parse specific month format: {time_period}, error: {e}")

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

        return first_of_previous.strftime("%Y-%m-%d"), last_of_previous.strftime("%Y-%m-%d")

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
        date_obj = current - timedelta(days=1)
        return date_obj.strftime("%Y-%m-%d"), date_obj.strftime("%Y-%m-%d")

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
                    "description": "Maximum number of rows to return (max 400)",
                },
            },
            "required": ["metrics", "dimensions", "start_date", "end_date"],
        },
    ),
    MCPTool(
        name="get_complex_analytics_report",
        description="Get complex multi-dimensional Adobe Analytics report with segment and dimension filtering",
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
                    "description": "List of dimensions for multi-level breakdown",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format",
                },
                "segment_filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "segment_id": {"type": "string"}
                        },
                        "required": ["segment_id"]
                    },
                    "description": "List of segment filters to apply"
                },
                "dimension_filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dimension": {"type": "string"},
                            "operator": {"type": "string", "enum": ["equals", "contains", "starts_with", "ends_with", "not_equals"]},
                            "values": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["dimension", "operator", "values"]
                    },
                    "description": "List of dimension filters to apply"
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Maximum number of rows to return (max 400)",
                },
                "sort_metric": {
                    "type": "string",
                    "description": "Metric to sort by"
                },
                "sort_direction": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "default": "desc",
                    "description": "Sort direction"
                }
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
        name="get_segments_list",
        description="Get list of available Adobe Analytics segments",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    MCPTool(
        name="get_dimension_values",
        description="Get available values for a specific dimension",
        input_schema={
            "type": "object",
            "properties": {
                "dimension": {
                    "type": "string",
                    "description": "Dimension to get values for",
                },
                "search_term": {
                    "type": "string",
                    "description": "Optional search term to filter values",
                },
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "description": "Maximum number of values to return",
                },
            },
            "required": ["dimension"],
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

@app.post("/mcp")
async def handle_mcp_request(request: MCPRequest) -> MCPResponse:
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

                return MCPResponse(
                    result={
                        "content": [
                            {"type": "text", "text": json.dumps(result, indent=2)}
                        ]
                    },
                    id=request.id,
                )

            elif tool_name == "get_complex_analytics_report":
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

                segment_filters = None
                if arguments.get("segment_filters"):
                    segment_filters = [
                        SegmentFilter(**sf) for sf in arguments["segment_filters"]
                    ]

                dimension_filters = None
                if arguments.get("dimension_filters"):
                    dimension_filters = [
                        DimensionFilter(**df) for df in arguments["dimension_filters"]
                    ]

                result = await get_complex_analytics_report(
                    metrics=arguments["metrics"],
                    dimensions=arguments["dimensions"],
                    start_date=arguments["start_date"],
                    end_date=arguments["end_date"],
                    segment_filters=segment_filters,
                    dimension_filters=dimension_filters,
                    limit=arguments.get("limit", 20),
                    sort_metric=arguments.get("sort_metric"),
                    sort_direction=arguments.get("sort_direction", "desc")
                )

                return MCPResponse(
                    result={
                        "content": [
                            {"type": "text", "text": json.dumps(result, indent=2)}
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

                return MCPResponse(
                    result={
                        "content": [
                            {"type": "text", "text": json.dumps(result, indent=2)}
                        ]
                    },
                    id=request.id,
                )

            elif tool_name == "get_segments_list":
                result = await get_segments_list()

                return MCPResponse(
                    result={
                        "content": [
                            {"type": "text", "text": json.dumps(result, indent=2)}
                        ]
                    },
                    id=request.id,
                )

            elif tool_name == "get_dimension_values":
                required_args = ["dimension"]
                missing_args = [arg for arg in required_args if arg not in arguments]
                if missing_args:
                    return MCPResponse(
                        error={
                            "code": -32602,
                            "message": f"Missing required arguments: {missing_args}",
                        },
                        id=request.id,
                    )

                result = await get_dimension_values(
                    dimension=arguments["dimension"],
                    search_term=arguments.get("search_term"),
                    limit=arguments.get("limit", 50)
                )

                return MCPResponse(
                    result={
                        "content": [
                            {"type": "text", "text": json.dumps(result, indent=2)}
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

                return MCPResponse(
                    result={
                        "content": [
                            {"type": "text", "text": json.dumps(result, indent=2)}
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
    return {
        "tools": [tool.dict() for tool in MCP_TOOLS],
        "server_type": "mcp_adobe_analytics",
        "total_tools": len(MCP_TOOLS),
    }

@app.on_event("startup")
async def startup_event():
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