"""
Predefined Adobe Analytics Schema
Based on Adobe Analytics Reporting API v2 official documentation
Contains commonly used metrics and dimensions for traffic, channel, and campaign analysis
"""

from typing import Dict, List, Optional
import difflib

# PREDEFINED METRICS - Based on Adobe Analytics API v2
PREDEFINED_METRICS = {
    # Traffic Metrics
    "metrics/visits": {
        "name": "Visits",
        "description": "Total number of visits to your site",
        "category": "traffic"
    },
    "metrics/visitors": {
        "name": "Unique Visitors", 
        "description": "Number of unique visitors",
        "category": "traffic"
    },
    "metrics/pageviews": {
        "name": "Page Views",
        "description": "Total number of page views",
        "category": "traffic"
    },
    "metrics/occurrences": {
        "name": "Occurrences",
        "description": "Number of hits where a dimension was set",
        "category": "traffic"
    },
    "metrics/bounces": {
        "name": "Bounces",
        "description": "Number of visits with only one page view",
        "category": "traffic"
    },
    "metrics/bouncerate": {
        "name": "Bounce Rate",
        "description": "Percentage of visits that bounced",
        "category": "traffic"
    },
    "metrics/entries": {
        "name": "Entries",
        "description": "Number of times a page was the first in a visit",
        "category": "traffic"
    },
    "metrics/exits": {
        "name": "Exits", 
        "description": "Number of times a page was the last in a visit",
        "category": "traffic"
    },
    "metrics/exitrate": {
        "name": "Exit Rate",
        "description": "Percentage of page views that were exits",
        "category": "traffic"
    },
    "metrics/singlepagevisits": {
        "name": "Single Page Visits",
        "description": "Visits with only one page view",
        "category": "traffic"
    },
    "metrics/reloads": {
        "name": "Reloads",
        "description": "Number of page reloads",
        "category": "traffic"
    },
    "metrics/mobileviews": {
        "name": "Mobile Views",
        "description": "Page views from mobile devices",
        "category": "mobile"
    },
    
    # Time-based Metrics
    "metrics/timespent": {
        "name": "Time Spent",
        "description": "Total time spent on site",
        "category": "engagement"
    },
    "metrics/averagetimespentonsite": {
        "name": "Average Time Spent on Site",
        "description": "Average time spent per visit",
        "category": "engagement"
    },
    "metrics/averagetimespentonpage": {
        "name": "Average Time Spent on Page",
        "description": "Average time spent per page view",
        "category": "engagement"
    },
    "metrics/averagevisitdepth": {
        "name": "Average Visit Depth",
        "description": "Average number of pages per visit",
        "category": "engagement"
    },
    
    # Conversion Metrics
    "metrics/orders": {
        "name": "Orders",
        "description": "Number of orders placed",
        "category": "conversion"
    },
    "metrics/revenue": {
        "name": "Revenue",
        "description": "Total revenue generated",
        "category": "conversion"
    },
    "metrics/units": {
        "name": "Units",
        "description": "Number of units sold",
        "category": "conversion"
    },
    "metrics/carts": {
        "name": "Carts",
        "description": "Number of shopping carts opened",
        "category": "conversion"
    },
    "metrics/cartadditions": {
        "name": "Cart Additions",
        "description": "Number of items added to cart",
        "category": "conversion"
    },
    "metrics/cartremovals": {
        "name": "Cart Removals", 
        "description": "Number of items removed from cart",
        "category": "conversion"
    },
    "metrics/cartviews": {
        "name": "Cart Views",
        "description": "Number of cart views",
        "category": "conversion"
    },
    "metrics/checkouts": {
        "name": "Checkouts",
        "description": "Number of checkout processes started",
        "category": "conversion"
    },
    "metrics/conversionrate": {
        "name": "Conversion Rate",
        "description": "Percentage of visits that converted",
        "category": "conversion"
    },
    
    # Campaign Metrics
    "metrics/campaigninstances": {
        "name": "Campaign Instances",
        "description": "Number of times campaign was set",
        "category": "campaign"
    },
    "metrics/clickthroughs": {
        "name": "Click-throughs",
        "description": "Number of click-throughs from campaigns",
        "category": "campaign"
    },
    "metrics/amo_impressions": {
        "name": "Adobe Advertising Impressions",
        "description": "Number of ad impressions",
        "category": "campaign"
    },
    "metrics/amo_clicks": {
        "name": "Adobe Advertising Clicks",
        "description": "Number of ad clicks",
        "category": "campaign"
    },
    "metrics/amo_cost": {
        "name": "Adobe Advertising Cost",
        "description": "Total cost of advertising",
        "category": "campaign"
    },
    
    # Search Metrics
    "metrics/searches": {
        "name": "Searches",
        "description": "Number of internal searches performed",
        "category": "search"
    },
    "metrics/searchresults": {
        "name": "Search Results",
        "description": "Number of search results returned",
        "category": "search"
    },
    
    # Video Metrics
    "metrics/videoviews": {
        "name": "Video Views",
        "description": "Number of video views",
        "category": "video"
    },
    "metrics/videocomplete": {
        "name": "Video Complete",
        "description": "Number of completed video views",
        "category": "video"
    },
    
    # Mobile App Metrics
    "metrics/mobileinstalls": {
        "name": "First Launches",
        "description": "Number of app installations",
        "category": "mobile"
    },
    "metrics/mobilelaunches": {
        "name": "Launches",
        "description": "Number of app launches",
        "category": "mobile"
    },
    "metrics/mobilecrashes": {
        "name": "Crashes",
        "description": "Number of app crashes",
        "category": "mobile"
    },
    "metrics/mobileupgrades": {
        "name": "Upgrades",
        "description": "Number of app upgrades",
        "category": "mobile"
    }
}

# PREDEFINED DIMENSIONS - Based on Adobe Analytics API v2
PREDEFINED_DIMENSIONS = {
    # Page Dimensions
    "variables/page": {
        "name": "Page",
        "description": "Page name or URL",
        "category": "page"
    },
    "variables/pagename": {
        "name": "Page Name",
        "description": "Friendly page name",
        "category": "page"
    },
    "variables/pageurl": {
        "name": "Page URL",
        "description": "Full page URL",
        "category": "page"
    },
    "variables/sitesection": {
        "name": "Site Section",
        "description": "Site section or category",
        "category": "page"
    },
    "variables/server": {
        "name": "Server",
        "description": "Web server name",
        "category": "page"
    },
    "variables/entrypage": {
        "name": "Entry Page",
        "description": "First page in visit",
        "category": "page"
    },
    "variables/exitpage": {
        "name": "Exit Page",
        "description": "Last page in visit",
        "category": "page"
    },
    "variables/pagesnotfound": {
        "name": "Pages Not Found",
        "description": "404 error pages",
        "category": "page"
    },
    
    # Traffic Source Dimensions
    "variables/referrer": {
        "name": "Referrer",
        "description": "Referring URL",
        "category": "traffic_source"
    },
    "variables/referrertype": {
        "name": "Referrer Type",
        "description": "Type of referrer (search engine, social, etc.)",
        "category": "traffic_source"
    },
    "variables/referringdomain": {
        "name": "Referring Domain",
        "description": "Domain of referring site",
        "category": "traffic_source"
    },
    "variables/searchengine": {
        "name": "Search Engine",
        "description": "Search engine used",
        "category": "traffic_source"
    },
    "variables/searchenginekeyword": {
        "name": "Search Engine Keyword",
        "description": "Search keywords used",
        "category": "traffic_source"
    },
    "variables/searchenginenatural": {
        "name": "Search Engine - Natural",
        "description": "Natural search engine traffic",
        "category": "traffic_source"
    },
    "variables/searchenginepaid": {
        "name": "Search Engine - Paid",
        "description": "Paid search engine traffic",
        "category": "traffic_source"
    },
    
    # Marketing Channel Dimensions
    "variables/marketingchannel": {
        "name": "Marketing Channel",
        "description": "Marketing channel classification",
        "category": "marketing"
    },
    "variables/marketingchanneldetail": {
        "name": "Marketing Channel Detail",
        "description": "Detailed marketing channel information",
        "category": "marketing"
    },
    "variables/lasttouchchannel": {
        "name": "Last Touch Channel",
        "description": "Last touch marketing channel",
        "category": "marketing"
    },
    "variables/firsttouchchannel": {
        "name": "First Touch Channel",
        "description": "First touch marketing channel",
        "category": "marketing"
    },
    "variables/lasttouchchanneldetail": {
        "name": "Last Touch Channel Detail",
        "description": "Last touch marketing channel details",
        "category": "marketing"
    },
    "variables/firsttouchchanneldetail": {
        "name": "First Touch Channel Detail",
        "description": "First touch marketing channel details",
        "category": "marketing"
    },
    
    # Campaign Dimensions
    "variables/campaign": {
        "name": "Tracking Code",
        "description": "Campaign tracking code",
        "category": "campaign"
    },
    "variables/campaign.utm-campaign": {
        "name": "utm_campaign",
        "description": "UTM campaign parameter",
        "category": "campaign"
    },
    "variables/campaign.utm-source": {
        "name": "utm_source",
        "description": "UTM source parameter",
        "category": "campaign"
    },
    "variables/campaign.utm-medium": {
        "name": "utm_medium",
        "description": "UTM medium parameter",
        "category": "campaign"
    },
    "variables/campaign.utm-content": {
        "name": "utm_content",
        "description": "UTM content parameter",
        "category": "campaign"
    },
    "variables/campaign.utm-term": {
        "name": "utm_term",
        "description": "UTM term parameter",
        "category": "campaign"
    },
    
    # Geographic Dimensions
    "variables/geocountry": {
        "name": "Country",
        "description": "Visitor country",
        "category": "geographic"
    },
    "variables/georegion": {
        "name": "Region",
        "description": "Visitor region/state",
        "category": "geographic"
    },
    "variables/geocity": {
        "name": "City",
        "description": "Visitor city",
        "category": "geographic"
    },
    "variables/geodma": {
        "name": "DMA",
        "description": "Designated Market Area",
        "category": "geographic"
    },
    "variables/language": {
        "name": "Language",
        "description": "Browser language setting",
        "category": "geographic"
    },
    "variables/zip": {
        "name": "Zip Code",
        "description": "Visitor zip/postal code",
        "category": "geographic"
    },
    
    # Technology Dimensions
    "variables/browser": {
        "name": "Browser",
        "description": "Web browser used",
        "category": "technology"
    },
    "variables/browsertype": {
        "name": "Browser Type",
        "description": "Browser type/family",
        "category": "technology"
    },
    "variables/browserversion": {
        "name": "Browser Version",
        "description": "Browser version number",
        "category": "technology"
    },
    "variables/operatingsystem": {
        "name": "Operating System",
        "description": "Operating system used",
        "category": "technology"
    },
    "variables/operatingsystemgroup": {
        "name": "Operating System Types",
        "description": "Operating system family",
        "category": "technology"
    },
    "variables/resolution": {
        "name": "Monitor Resolution",
        "description": "Screen resolution",
        "category": "technology"
    },
    "variables/colordepth": {
        "name": "Color Depth",
        "description": "Monitor color depth",
        "category": "technology"
    },
    "variables/javascriptenabled": {
        "name": "JavaScript Enabled",
        "description": "JavaScript support status",
        "category": "technology"
    },
    "variables/javaenabled": {
        "name": "Java Enabled",
        "description": "Java support status",
        "category": "technology"
    },
    "variables/cookie": {
        "name": "Cookie Support",
        "description": "Cookie support status",
        "category": "technology"
    },
    "variables/connectiontype": {
        "name": "Connection Type",
        "description": "Internet connection type",
        "category": "technology"
    },
    
    # Mobile Dimensions
    "variables/mobiledevicetype": {
        "name": "Mobile Device Type",
        "description": "Type of mobile device",
        "category": "mobile"
    },
    "variables/mobiledevicename": {
        "name": "Mobile Device",
        "description": "Mobile device model name",
        "category": "mobile"
    },
    "variables/mobilecarrier": {
        "name": "Mobile Carrier",
        "description": "Mobile service carrier",
        "category": "mobile"
    },
    "variables/mobilescreensize": {
        "name": "Mobile Screen Size",
        "description": "Mobile device screen size",
        "category": "mobile"
    },
    "variables/mobilemanufacturer": {
        "name": "Mobile Manufacturer",
        "description": "Mobile device manufacturer",
        "category": "mobile"
    },
    
    # Time Dimensions
    "variables/daterangeday": {
        "name": "Day",
        "description": "Day of the month",
        "category": "time"
    },
    "variables/daterangeweek": {
        "name": "Week",
        "description": "Week of the year",
        "category": "time"
    },
    "variables/daterangemonth": {
        "name": "Month",
        "description": "Month of the year",
        "category": "time"
    },
    "variables/daterangequarter": {
        "name": "Quarter",
        "description": "Quarter of the year",
        "category": "time"
    },
    "variables/daterangeyear": {
        "name": "Year",
        "description": "Year",
        "category": "time"
    },
    "variables/daterangehour": {
        "name": "Hour",
        "description": "Hour of the day",
        "category": "time"
    },
    "variables/dayofweek": {
        "name": "Day of Week",
        "description": "Day of the week",
        "category": "time"
    },
    "variables/timepartdayofweek": {
        "name": "Day of Week",
        "description": "Day of the week (Sunday-Saturday)",
        "category": "time"
    },
    "variables/timeparthourofday": {
        "name": "Hour of Day",
        "description": "Hour of the day (0-23)",
        "category": "time"
    },
    
    # Visit Dimensions
    "variables/visitnumber": {
        "name": "Visit Number",
        "description": "Visit number for the visitor",
        "category": "visit"
    },
    "variables/visitorid": {
        "name": "Visitor ID",
        "description": "Unique visitor identifier",
        "category": "visit"
    },
    "variables/customerid": {
        "name": "Customer ID",
        "description": "Customer identifier",
        "category": "visit"
    },
    "variables/newrepeat": {
        "name": "New/Repeat Visitor",
        "description": "New vs repeat visitor classification",
        "category": "visit"
    },
    "variables/returnfrequency": {
        "name": "Return Frequency",
        "description": "How often visitor returns",
        "category": "visit"
    },
    "variables/dayssincelastvisit": {
        "name": "Days Since Last Visit",
        "description": "Days since last visit",
        "category": "visit"
    },
    "variables/hitdepth": {
        "name": "Hit Depth",
        "description": "Depth of hit in visit",
        "category": "visit"
    },
    "variables/pathlength": {
        "name": "Visit Depth",
        "description": "Number of pages in visit",
        "category": "visit"
    },
    
    # Audience Dimensions
    "variables/mcaudiences": {
        "name": "Audiences ID",
        "description": "Experience Cloud audience IDs",
        "category": "audience"
    },
    "variables/mcvisid": {
        "name": "Experience Cloud Visitor ID",
        "description": "Experience Cloud visitor identifier",
        "category": "audience"
    },
    
    # Custom Link Dimensions
    "variables/customlink": {
        "name": "Custom Link",
        "description": "Custom link name",
        "category": "links"
    },
    "variables/downloadlink": {
        "name": "Download Link",
        "description": "Download link name",
        "category": "links"
    },
    "variables/exitlink": {
        "name": "Exit Link",
        "description": "Exit link name",
        "category": "links"
    }
}

# Analysis Templates for Common Use Cases
ANALYSIS_TEMPLATES = {
    "traffic_overview": {
        "name": "Traffic Overview",
        "description": "Basic traffic analysis with page breakdown",
        "metrics": ["metrics/visits", "metrics/pageviews", "metrics/visitors"],
        "dimensions": ["variables/page"],
        "use_case": "Understanding overall site traffic patterns"
    },
    "channel_performance": {
        "name": "Marketing Channel Performance",
        "description": "Analysis of marketing channel effectiveness",
        "metrics": ["metrics/visits", "metrics/orders", "metrics/revenue"],
        "dimensions": ["variables/marketingchannel"],
        "use_case": "Evaluating marketing channel ROI"
    },
    "campaign_analysis": {
        "name": "Campaign Analysis",
        "description": "Campaign performance tracking",
        "metrics": ["metrics/visits", "metrics/clickthroughs", "metrics/conversionrate"],
        "dimensions": ["variables/campaign"],
        "use_case": "Measuring campaign effectiveness"
    },
    "content_performance": {
        "name": "Content Performance",
        "description": "Page-level content analysis",
        "metrics": ["metrics/pageviews", "metrics/averagetimespentonpage", "metrics/bouncerate"],
        "dimensions": ["variables/page"],
        "use_case": "Identifying top and underperforming content"
    },
    "device_analysis": {
        "name": "Device Analysis",
        "description": "Mobile vs desktop performance",
        "metrics": ["metrics/visits", "metrics/conversionrate", "metrics/bouncerate"],
        "dimensions": ["variables/mobiledevicetype"],
        "use_case": "Understanding device-specific user behavior"
    },
    "geographic_analysis": {
        "name": "Geographic Analysis",
        "description": "Location-based performance insights",
        "metrics": ["metrics/visits", "metrics/revenue", "metrics/conversionrate"],
        "dimensions": ["variables/geocountry"],
        "use_case": "Analyzing performance by geographic region"
    },
    "search_analysis": {
        "name": "Search Analysis",
        "description": "Analysis of search engine traffic",
        "metrics": ["metrics/visits", "metrics/bouncerate", "metrics/conversionrate"],
        "dimensions": ["variables/searchengine", "variables/searchenginekeyword"],
        "use_case": "Optimizing search engine marketing"
    },
    "entry_exit_analysis": {
        "name": "Entry/Exit Analysis",
        "description": "Analysis of entry and exit points",
        "metrics": ["metrics/entries", "metrics/exits", "metrics/bouncerate"],
        "dimensions": ["variables/entrypage", "variables/exitpage"],
        "use_case": "Improving site entry and exit points"
    },
    "technology_analysis": {
        "name": "Technology Analysis",
        "description": "Analysis of visitor technology",
        "metrics": ["metrics/visits", "metrics/pageviews"],
        "dimensions": ["variables/browser", "variables/operatingsystem"],
        "use_case": "Optimizing for different technologies"
    },
    "visitor_retention": {
        "name": "Visitor Retention",
        "description": "Analysis of visitor return patterns",
        "metrics": ["metrics/visits", "metrics/visitors"],
        "dimensions": ["variables/visitnumber", "variables/returnfrequency"],
        "use_case": "Understanding visitor loyalty"
    }
}

def get_all_metrics() -> List[str]:
    """Get list of all available metric IDs"""
    return list(PREDEFINED_METRICS.keys())

def get_all_dimensions() -> List[str]:
    """Get list of all available dimension IDs"""
    return list(PREDEFINED_DIMENSIONS.keys())

def get_metrics_by_category(category: str) -> List[str]:
    """Get metrics filtered by category"""
    return [
        metric_id for metric_id, info in PREDEFINED_METRICS.items()
        if info["category"] == category
    ]

def get_dimensions_by_category(category: str) -> List[str]:
    """Get dimensions filtered by category"""
    return [
        dim_id for dim_id, info in PREDEFINED_DIMENSIONS.items()
        if info["category"] == category
    ]

def validate_metric(metric_id: str) -> Optional[str]:
    """
    Validate and find best match for metric ID
    Returns the valid metric ID or None if not found
    """
    # Direct match
    if metric_id in PREDEFINED_METRICS:
        return metric_id
    
    # Fuzzy matching
    all_metrics = get_all_metrics()
    matches = difflib.get_close_matches(metric_id.lower(), 
                                       [m.lower() for m in all_metrics], 
                                       n=1, cutoff=0.6)
    if matches:
        # Find the original case metric
        for metric in all_metrics:
            if metric.lower() == matches[0]:
                return metric
    
    # Search in names and descriptions
    search_term = metric_id.lower()
    for metric_id_key, info in PREDEFINED_METRICS.items():
        if (search_term in info["name"].lower() or 
            search_term in info["description"].lower()):
            return metric_id_key
    
    return None

def validate_dimension(dimension_id: str) -> Optional[str]:
    """
    Validate and find best match for dimension ID
    Returns the valid dimension ID or None if not found
    """
    # Direct match
    if dimension_id in PREDEFINED_DIMENSIONS:
        return dimension_id
    
    # Fuzzy matching
    all_dimensions = get_all_dimensions()
    matches = difflib.get_close_matches(dimension_id.lower(),
                                       [d.lower() for d in all_dimensions],
                                       n=1, cutoff=0.6)
    if matches:
        # Find the original case dimension
        for dimension in all_dimensions:
            if dimension.lower() == matches[0]:
                return dimension
    
    # Search in names and descriptions
    search_term = dimension_id.lower()
    for dim_id_key, info in PREDEFINED_DIMENSIONS.items():
        if (search_term in info["name"].lower() or 
            search_term in info["description"].lower()):
            return dim_id_key
    
    return None

def get_metric_info(metric_id: str) -> Optional[Dict]:
    """Get detailed information about a metric"""
    return PREDEFINED_METRICS.get(metric_id)

def get_dimension_info(dimension_id: str) -> Optional[Dict]:
    """Get detailed information about a dimension"""
    return PREDEFINED_DIMENSIONS.get(dimension_id)

def get_analysis_template(template_name: str) -> Optional[Dict]:
    """Get predefined analysis template"""
    return ANALYSIS_TEMPLATES.get(template_name)

def get_all_analysis_templates() -> Dict[str, Dict]:
    """Get all available analysis templates"""
    return ANALYSIS_TEMPLATES

def search_metrics(query: str) -> List[Dict]:
    """Search metrics by name or description"""
    results = []
    query_lower = query.lower()
    
    for metric_id, info in PREDEFINED_METRICS.items():
        if (query_lower in info["name"].lower() or 
            query_lower in info["description"].lower() or
            query_lower in metric_id.lower()):
            results.append({
                "id": metric_id,
                "name": info["name"],
                "description": info["description"],
                "category": info["category"]
            })
    
    return results

def search_dimensions(query: str) -> List[Dict]:
    """Search dimensions by name or description"""
    results = []
    query_lower = query.lower()
    
    for dim_id, info in PREDEFINED_DIMENSIONS.items():
        if (query_lower in info["name"].lower() or 
            query_lower in info["description"].lower() or
            query_lower in dim_id.lower()):
            results.append({
                "id": dim_id,
                "name": info["name"],
                "description": info["description"],
                "category": info["category"]
            })
    
    return results

# Export lists for backward compatibility
METRICS = get_all_metrics()
DIMENSIONS = get_all_dimensions()