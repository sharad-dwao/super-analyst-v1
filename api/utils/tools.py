import requests
import difflib
import json
import logging
import os
from datetime import date
from typing import List, Dict, Union, Optional
from dotenv import load_dotenv
from .adobe_analytics_schema import (
    validate_metric, 
    validate_dimension, 
    get_all_metrics, 
    get_all_dimensions,
    METRICS,
    DIMENSIONS
)

load_dotenv(".env.local")

# Configure logging
tlogging_format = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO, format=tlogging_format, datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

CLIENT_ID = os.environ.get("ADOBE_CLIENT_ID")
CLIENT_SECRET = os.environ.get("ADOBE_CLIENT_SECRET")
COMPANY_ID = os.environ.get("ADOBE_COMPANY_ID")
ORG_ID = os.environ.get("ADOBE_ORG_ID")
REPORTSUIT_ID = os.environ.get("ADOBE_REPORTSUIT_ID")


def get_current_date() -> dict:
    return {"date": date.today().isoformat()}


def get_access_token() -> str:
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


def get_report_adobe_analytics(
    metrics: Union[List[Dict[str, str]], Dict[str, str], str, List[str]],
    dimension: str,
    start_date: str,
    end_date: str,
    dimension2: Optional[str] = None,
    limit: int = 20,
):
    """
    Fetch Adobe Analytics report with support for 1 or 2 dimensions
    Uses predefined schema for validation

    Args:
        metrics: Metric(s) to include in the report
        dimension: Primary dimension for breakdown
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        dimension2: Optional second dimension for cross-dimensional analysis
        limit: Maximum number of rows to return (default: 20)

    Returns:
        Dict containing the report data or error information
    """
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json;charset=utf-8",
        "x-gw-ims-org-id": ORG_ID,
        "x-proxy-global-company-id": COMPANY_ID,
        "x-api-key": CLIENT_ID,
        "Authorization": f"Bearer {get_access_token()}",
    }

    # Normalize metrics into list of dicts
    if isinstance(metrics, str):
        metrics = [{"id": metrics}]
    elif isinstance(metrics, list):
        # Handle list of strings or dicts
        normalized_metrics = []
        for m in metrics:
            if isinstance(m, str):
                normalized_metrics.append({"id": m})
            elif isinstance(m, dict):
                normalized_metrics.append(m)
        metrics = normalized_metrics
    elif isinstance(metrics, dict):
        metrics = [metrics]

    # Validate and prepare metrics using predefined schema
    metric_entries = []
    for idx, m in enumerate(metrics):
        if isinstance(m, str):
            m = {"id": m}
        
        # Use predefined schema validation
        validated_metric = validate_metric(m["id"])
        if validated_metric:
            metric_entries.append({"columnId": str(idx), "id": validated_metric})
            logger.info(f"Validated metric: {m['id']} -> {validated_metric}")
        else:
            logger.warning(f"Metric '{m['id']}' not found in predefined schema, using as-is")
            metric_entries.append({"columnId": str(idx), "id": m["id"]})

    # Validate and prepare primary dimension using predefined schema
    validated_dimension = validate_dimension(dimension)
    if validated_dimension:
        valid_dim = validated_dimension
        logger.info(f"Validated dimension: {dimension} -> {validated_dimension}")
    else:
        logger.warning(f"Primary dimension '{dimension}' not found in predefined schema, using as-is")
        valid_dim = dimension

    # Initialize valid_dim2 to None
    valid_dim2 = None

    # Handle second dimension for cross-dimensional analysis
    if dimension2:
        validated_dimension2 = validate_dimension(dimension2)
        if validated_dimension2:
            valid_dim2 = validated_dimension2
            logger.info(f"Validated secondary dimension: {dimension2} -> {validated_dimension2}")
        else:
            logger.warning(f"Secondary dimension '{dimension2}' not found in predefined schema, using as-is")
            valid_dim2 = dimension2

    # Special handling for date-related dimensions
    is_time_dimension = False
    time_dimension_id = None
    
    # Check if primary dimension is time-related
    if valid_dim and valid_dim.startswith("variables/daterange"):
        is_time_dimension = True
        time_dimension_id = valid_dim
        
        # For time dimensions, we need to use a different approach
        # We'll use the dateRange in globalFilters and set the granularity
        granularity = "day"  # Default granularity
        
        if "month" in valid_dim:
            granularity = "month"
        elif "week" in valid_dim:
            granularity = "week"
        elif "quarter" in valid_dim:
            granularity = "quarter"
        elif "year" in valid_dim:
            granularity = "year"
        elif "hour" in valid_dim:
            granularity = "hour"
            
        logger.info(f"Using time dimension with granularity: {granularity}")
        
        # For time dimensions, we'll use a different dimension in the request
        # and handle the time aspect through the dateRange
        valid_dim = "variables/daterangeday"  # Use a generic time dimension

    # Build the request body
    body = {
        "rsid": REPORTSUIT_ID,
        "globalFilters": [
            {
                "type": "dateRange",
                "dateRange": f"{start_date}T00:00:00/{end_date}T23:59:59",
            }
        ],
        "metricContainer": {"metrics": metric_entries},
        "dimension": valid_dim,
        "settings": {
            "limit": limit,
            "page": 0,
            "dimensionSort": "asc",
            "countRepeatInstances": True,
        },
    }
    
    # Add time granularity if using a time dimension
    if is_time_dimension:
        if "granularity" not in body:
            body["settings"]["dimensionSort"] = "asc"
            
        # Add time info to the request
        if time_dimension_id == "variables/daterangemonth":
            body["settings"]["dimensionSort"] = "asc"
            # Use the time dimension's granularity
            body["globalFilters"].append({
                "type": "dateRangeGranularity",
                "granularity": "month"
            })
        elif time_dimension_id == "variables/daterangeweek":
            body["settings"]["dimensionSort"] = "asc"
            body["globalFilters"].append({
                "type": "dateRangeGranularity",
                "granularity": "week"
            })
        elif time_dimension_id == "variables/daterangequarter":
            body["settings"]["dimensionSort"] = "asc"
            body["globalFilters"].append({
                "type": "dateRangeGranularity",
                "granularity": "quarter"
            })
        elif time_dimension_id == "variables/daterangeyear":
            body["settings"]["dimensionSort"] = "asc"
            body["globalFilters"].append({
                "type": "dateRangeGranularity",
                "granularity": "year"
            })
        elif time_dimension_id == "variables/daterangehour":
            body["settings"]["dimensionSort"] = "asc"
            body["globalFilters"].append({
                "type": "dateRangeGranularity",
                "granularity": "hour"
            })

    # Add second dimension configuration if provided
    if valid_dim2:
        # For Adobe Analytics 2.0 API, we use metricsFilters for cross-dimensional analysis
        # This approach creates a breakdown report
        body["metricContainer"]["metricFilters"] = [
            {
                "id": "0",
                "type": "breakdown",
                "dimension": valid_dim2,
                "itemId": "*",  # Get all items for breakdown
            }
        ]

        # Adjust settings for two-dimensional reports
        body["settings"]["limit"] = min(limit, 50)  # Reduce limit for breakdown reports
        body["settings"]["includeAnomalyDetection"] = False

        logger.info(
            f"Using two-dimensional analysis: {valid_dim} with breakdown by {valid_dim2}"
        )
    else:
        logger.info(f"Using single-dimensional analysis: {valid_dim}")

    url = f"https://analytics.adobe.io/api/{COMPANY_ID}/reports"

    try:
        logger.info(
            f"Making Adobe Analytics request with body: {json.dumps(body, indent=2)}"
        )
        res = requests.post(url, headers=headers, json=body, timeout=30)
        res.raise_for_status()

        response_data = res.json()

        # Process and enhance the response data
        processed_data = _process_analytics_response(
            response_data, valid_dim, valid_dim2, metric_entries
        )

        # Add metadata about the request for better understanding
        metadata = {
            "request_type": "two_dimensional" if valid_dim2 else "single_dimensional",
            "primary_dimension": valid_dim,
            "secondary_dimension": valid_dim2,
            "metrics": [m["id"] for m in metric_entries],
            "date_range": f"{start_date} to {end_date}",
            "total_rows": len(response_data.get("rows", [])),
            "limit_applied": limit,
            "api_version": "2.0",
            "schema_validation": "predefined_schema_used",
            "is_time_dimension": is_time_dimension,
            "time_dimension_id": time_dimension_id
        }

        return {
            "metadata": metadata,
            "result": response_data,
            "processed_data": processed_data,
            "debug": {"headers": headers, "body": body},
        }

    except requests.exceptions.HTTPError as err:
        error_details = {
            "error": f"{err.response.status_code} {err.response.reason}",
            "details": err.response.text,
            "request_body": body,
        }
        logger.error(f"Adobe Analytics API error: {error_details}")
        return error_details
    except Exception as e:
        error_details = {"error": f"Unexpected error: {str(e)}", "request_body": body}
        logger.error(f"Unexpected error in Adobe Analytics request: {error_details}")
        return error_details


def _process_analytics_response(
    response_data: Dict,
    primary_dim: str,
    secondary_dim: Optional[str],
    metrics: List[Dict],
) -> Dict:
    """
    Process and structure the Adobe Analytics response for better readability
    """
    try:
        processed = {
            "summary": {
                "total_rows": len(response_data.get("rows", [])),
                "dimensions": [primary_dim],
                "metrics": [m["id"] for m in metrics],
                "has_breakdown": secondary_dim is not None,
            },
            "data": [],
        }

        if secondary_dim:
            processed["summary"]["dimensions"].append(secondary_dim)

        # Process rows
        for row in response_data.get("rows", []):
            row_data = {"dimension_values": {}, "metric_values": {}}

            # Extract dimension values
            if "value" in row:
                row_data["dimension_values"][primary_dim] = row["value"]

            # Extract metric values
            if "data" in row:
                for idx, metric_value in enumerate(row["data"]):
                    if idx < len(metrics):
                        metric_id = metrics[idx]["id"]
                        row_data["metric_values"][metric_id] = metric_value

            # Handle breakdown data if present
            if "breakdown" in row and secondary_dim:
                row_data["breakdown"] = []
                for breakdown_item in row["breakdown"]:
                    breakdown_data = {
                        "dimension_values": {
                            secondary_dim: breakdown_item.get("value", "")
                        },
                        "metric_values": {},
                    }
                    if "data" in breakdown_item:
                        for idx, metric_value in enumerate(breakdown_item["data"]):
                            if idx < len(metrics):
                                metric_id = metrics[idx]["id"]
                                breakdown_data["metric_values"][
                                    metric_id
                                ] = metric_value
                    row_data["breakdown"].append(breakdown_data)

            processed["data"].append(row_data)

        return processed

    except Exception as e:
        logger.error(f"Error processing analytics response: {str(e)}")
        return {"summary": {"error": f"Processing error: {str(e)}"}, "data": []}


def get_report_adobe_analytics_multi_dimension(
    metrics: Union[List[str], str],
    dimensions: List[str],
    start_date: str,
    end_date: str,
    limit: int = 20,
):
    """
    Specialized function for multi-dimensional Adobe Analytics reports

    Args:
        metrics: Metric(s) to include in the report
        dimensions: List of dimensions (up to 2 supported)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        limit: Maximum number of rows to return

    Returns:
        Dict containing the report data
    """
    if len(dimensions) == 0:
        raise ValueError("At least one dimension is required")
    elif len(dimensions) == 1:
        return get_report_adobe_analytics(
            metrics=metrics,
            dimension=dimensions[0],
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
    elif len(dimensions) == 2:
        return get_report_adobe_analytics(
            metrics=metrics,
            dimension=dimensions[0],
            dimension2=dimensions[1],
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
    else:
        logger.warning(
            f"More than 2 dimensions requested ({len(dimensions)}), using first 2"
        )
        return get_report_adobe_analytics(
            metrics=metrics,
            dimension=dimensions[0],
            dimension2=dimensions[1],
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )


# Export the predefined schema lists
logger.info(f"Loaded predefined schema: {len(METRICS)} metrics, {len(DIMENSIONS)} dimensions")