"""
MCP-based Analytics Pipeline
Replaces LangChain pipeline with MCP server integration
SECURITY: Only communicates with internal MCP servers
"""

import json
import logging
import os
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from .mcp_client import AdobeAnalyticsMCPClient, MCPTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AnalyticsQuery(BaseModel):
    """Structured analytics query after enhancement"""
    enhanced_query: str = Field(description="Enhanced and clarified user query")
    intent: str = Field(description="Primary intent of the query")
    metrics: List[str] = Field(description="List of relevant metrics to analyze")
    dimensions: List[str] = Field(description="List of relevant dimensions")
    time_period: str = Field(description="Time period for analysis")
    comparison_period: str = Field(description="Optional comparison period", default="")
    output_format: str = Field(description="Preferred output format", default="detailed")
    additional_context: str = Field(description="Additional context or requirements")


class AnalyticsResult(BaseModel):
    """Final analytics result with insights"""
    summary: str = Field(description="Executive summary of findings")
    key_insights: List[str] = Field(description="List of key insights")
    data_analysis: str = Field(description="Detailed analysis of the data")
    recommendations: List[str] = Field(description="Actionable recommendations")
    raw_data: Dict[str, Any] = Field(description="Raw data from analytics")


class MCPAnalyticsPipeline:
    """Analytics pipeline using MCP servers"""
    
    def __init__(self, openai_api_key: str, mcp_server_url: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.llm = ChatOpenAI(
            model="openai/gpt-4.1-mini",
            api_key=openai_api_key,
            base_url=base_url,
            temperature=0.1,
            streaming=True
        )
        
        # SECURITY: Validate MCP server URL is internal
        if not self._is_internal_url(mcp_server_url):
            raise ValueError(f"MCP server URL must be internal: {mcp_server_url}")
        
        self.mcp_client = AdobeAnalyticsMCPClient(mcp_server_url)
        self.available_tools = []
        self.mcp_server_url = mcp_server_url
        
    def _is_internal_url(self, url: str) -> bool:
        """Validate that URL is internal/localhost only"""
        internal_hosts = ["localhost", "127.0.0.1", "::1"]
        return any(host in url for host in internal_hosts)
        
    async def initialize(self):
        """Initialize the pipeline and discover available tools"""
        try:
            # Check MCP server health first
            health = await self.mcp_client.health_check()
            if health.get("status") != "healthy":
                logger.warning(f"MCP server health check failed: {health}")
            
            # Discover available tools
            self.available_tools = await self.mcp_client.list_available_tools()
            logger.info(f"Initialized MCP pipeline with {len(self.available_tools)} tools")
            
            if not self.available_tools:
                logger.warning("No tools discovered from MCP server")
                
        except Exception as e:
            logger.error(f"Failed to initialize MCP pipeline: {str(e)}")
            self.available_tools = []
    
    def _get_enhancement_system_prompt(self) -> str:
        """Get system prompt for query enhancement with MCP tool context"""
        tools_info = ""
        for tool in self.available_tools:
            tools_info += f"- {tool.name}: {tool.description}\n"
        
        if not tools_info:
            tools_info = "- get_analytics_report: Get Adobe Analytics report\n- get_comparison_report: Compare data between periods\n"
        
        return f"""You are an expert analytics query enhancer using MCP (Model Context Protocol) servers for Adobe Analytics. Your job is to take user queries about web analytics and enhance them for better analysis.

## Available MCP Tools:
{tools_info}

## Common Adobe Analytics Metrics:
- metrics/visits: Total visits to the site
- metrics/visitors: Unique visitors
- metrics/pageviews: Total page views
- metrics/bounces: Single-page visits
- metrics/bouncerate: Percentage of bounced visits
- metrics/orders: Number of orders
- metrics/revenue: Total revenue
- metrics/conversionrate: Conversion percentage

## Common Adobe Analytics Dimensions:
- variables/page: Page name or URL
- variables/referrer: Referring URL
- variables/campaign: Campaign tracking code
- variables/geocountry: Visitor country
- variables/browser: Web browser
- variables/mobiledevicetype: Mobile device type
- variables/marketingchannel: Marketing channel
- variables/daterangemonth: Monthly breakdown
- variables/daterangeweek: Weekly breakdown
- variables/daterangeday: Daily breakdown

## OUTPUT FORMAT DETECTION:
Pay special attention to how the user wants the output formatted:
- "summary" / "summarize" / "brief" → output_format: "summary"
- "detailed" / "in detail" / "comprehensive" → output_format: "detailed"
- "table" / "tabular" / "in a table" → output_format: "table"
- "chart" / "graph" / "visual" → output_format: "chart"
- "executive summary" / "high-level" → output_format: "executive"
- "bullet points" / "list" → output_format: "list"
- "quick overview" / "at a glance" → output_format: "brief"

If no specific format is mentioned, default to "detailed" for comprehensive analysis.

## COMPARISON ANALYSIS HANDLING:
When users ask for comparisons between time periods:
- Set the primary time_period to the most recent period
- Set comparison_period to the earlier period for comparison
- Use specific month names when possible (e.g., "november_2024", "october_2024")

## TIME PERIOD SPECIFICATIONS:
- For "this month" use: "current_month" or specific month like "november_2024"
- For "last month" use: "previous_month" or specific month like "october_2024"
- For "month over month" use both periods

Your task:
1. Understand the user's intent and clarify ambiguous requests
2. Identify the most relevant metrics and dimensions from the available schema
3. Determine appropriate time periods if not specified
4. **DETECT THE PREFERRED OUTPUT FORMAT** from user language
5. For comparisons, identify both primary and comparison periods
6. Enhance the query with analytics best practices

Guidelines:
- Use only valid Adobe Analytics metrics and dimensions
- Limit dimensions to maximum 2 for optimal performance
- Choose metrics that directly answer the user's question
- For time-based analysis, consider using date range dimensions

You must respond with a valid JSON object with these exact fields:
- enhanced_query: string
- intent: string
- metrics: array of strings (use valid Adobe Analytics metric IDs)
- dimensions: array of strings (max 2 elements, use valid dimension IDs)
- time_period: string
- comparison_period: string (empty if no comparison needed)
- output_format: string (detected from user query or "detailed" as default)
- additional_context: string

Respond with valid JSON only, no additional text or formatting."""

    def _get_analysis_system_prompt(self) -> str:
        """Get system prompt for data analysis"""
        return """You are an expert data analyst specializing in web analytics using MCP servers. Your job is to analyze Adobe Analytics data and provide actionable insights in the format preferred by the user.

## OUTPUT FORMAT GUIDELINES:

**CRITICAL: Always adapt your response format based on the user's preference:**

1. **"summary" or "brief"**: Provide concise, high-level findings in 2-3 sentences per section
2. **"detailed"**: Provide comprehensive analysis with full explanations (DEFAULT)
3. **"table"**: Structure key findings in markdown table format
4. **"chart"**: Describe data in a way that suggests visual representation
5. **"executive"**: Focus on business impact and strategic insights
6. **"list"**: Use bullet points and numbered lists extensively

## RESPONSE STRUCTURE BY FORMAT:

### For "summary" or "brief":
- summary: 2-3 sentences maximum
- key_insights: 3-5 bullet points, each 1 sentence
- data_analysis: 1-2 paragraphs maximum
- recommendations: 3-4 actionable items, each 1 sentence

### For "detailed" (default):
- summary: Comprehensive executive summary (3-4 sentences)
- key_insights: 5-8 detailed bullet points with context
- data_analysis: Full analysis with explanations, trends, and context
- recommendations: 5-7 detailed, actionable recommendations with rationale

### For "table":
- summary: Brief overview mentioning table format
- key_insights: Present as markdown table when possible
- data_analysis: Include data tables and structured comparisons
- recommendations: Numbered list with clear priorities

### For "executive":
- summary: Strategic overview focusing on business impact
- key_insights: Business-focused insights with revenue/growth implications
- data_analysis: Focus on trends affecting business objectives
- recommendations: Strategic actions with business justification

### For "list":
- summary: Brief overview in bullet format
- key_insights: Numbered or bulleted list
- data_analysis: Structured with clear headings and sub-points
- recommendations: Prioritized numbered list

## Analysis Guidelines:
1. **FIRST: Identify the requested output format and adapt accordingly**
2. Analyze the raw data from MCP servers
3. Identify key patterns, trends, and anomalies
4. Provide clear, actionable insights in the preferred format
5. Make data-driven recommendations
6. Present findings in a business-friendly format

## Multi-Dimensional Analysis:
- When analyzing cross-dimensional data, look for interaction patterns
- Identify which combinations perform best/worst
- Compare performance across different dimension values
- Look for opportunities in underperforming segments

## Comparison Analysis:
- When comparing time periods, calculate percentage changes and growth rates
- Identify significant trends and changes between periods
- Highlight both positive and negative changes
- Provide context for why changes might have occurred
- Focus on actionable insights from the comparison

## Error Handling:
- If data is incomplete or has errors, acknowledge this in the analysis
- Provide recommendations based on available data
- Suggest follow-up analyses if needed

**REMEMBER: The output format preference is CRITICAL - users expect their specified format to be respected.**

You must respond with a valid JSON object with these exact fields:
- summary: string (formatted according to user preference)
- key_insights: array of strings (formatted according to user preference)
- data_analysis: string (formatted according to user preference)
- recommendations: array of strings (formatted according to user preference)

Respond with valid JSON only, no additional text or formatting."""

    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query through the MCP-based pipeline"""
        try:
            logger.info(f"Starting MCP-based analysis for query: {user_query}")
            
            # Initialize if not already done
            if not self.available_tools:
                await self.initialize()
            
            # Stage 1: Query Enhancement
            logger.info("Stage 1: Enhancing query...")
            enhanced_query_data = await self._enhance_query(user_query)
            
            # Stage 2: Data Retrieval via MCP
            logger.info("Stage 2: Retrieving analytics data via MCP...")
            raw_data = await self._get_analytics_data_mcp(enhanced_query_data)
            
            # Stage 3: Analysis and Insights
            logger.info("Stage 3: Analyzing data and generating insights...")
            final_result = await self._analyze_data(enhanced_query_data, raw_data)
            
            return {
                "stage_1_enhancement": enhanced_query_data.dict() if hasattr(enhanced_query_data, 'dict') else enhanced_query_data,
                "stage_2_raw_data": raw_data,
                "stage_3_analysis": final_result.dict() if hasattr(final_result, 'dict') else final_result,
                "success": True,
                "mcp_info": {
                    "server_url": self.mcp_server_url,
                    "tools_available": len(self.available_tools),
                    "server_type": "mcp_adobe_analytics",
                    "pipeline_version": "mcp_v1.0"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in MCP pipeline: {str(e)}")
            return {
                "error": str(e),
                "success": False,
                "mcp_info": {
                    "server_url": self.mcp_server_url,
                    "tools_available": len(self.available_tools),
                    "server_type": "mcp_adobe_analytics"
                }
            }

    async def _enhance_query(self, user_query: str) -> AnalyticsQuery:
        """Stage 1: Enhance the user query"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_enhancement_system_prompt()),
                ("human", "User Query: {user_query}")
            ])
            
            chain = prompt | self.llm
            response = await chain.ainvoke({"user_query": user_query})
            
            content = response.content.strip()
            logger.info(f"Enhancement response: {content}")
            
            result_dict = self._extract_json_from_response(content)
            
            # Validate and clean the result
            result_dict = self._validate_enhancement_result(result_dict)
            
            return AnalyticsQuery(**result_dict)
                
        except Exception as e:
            logger.error(f"Error in query enhancement: {str(e)}")
            return AnalyticsQuery(
                enhanced_query=user_query,
                intent="general_analysis",
                metrics=["metrics/visits"],
                dimensions=["variables/page"],
                time_period="yesterday",
                comparison_period="",
                output_format="detailed",
                additional_context="Fallback enhancement due to processing error"
            )

    def _validate_enhancement_result(self, result_dict: Dict) -> Dict:
        """Validate and clean enhancement result"""
        # Ensure required fields exist
        required_fields = ["enhanced_query", "intent", "metrics", "dimensions", "time_period", "output_format"]
        for field in required_fields:
            if field not in result_dict:
                logger.warning(f"Missing field {field} in enhancement result")
                if field == "metrics":
                    result_dict[field] = ["metrics/visits"]
                elif field == "dimensions":
                    result_dict[field] = ["variables/page"]
                elif field == "output_format":
                    result_dict[field] = "detailed"
                else:
                    result_dict[field] = ""
        
        # Ensure metrics and dimensions are lists
        if not isinstance(result_dict["metrics"], list):
            result_dict["metrics"] = [result_dict["metrics"]] if result_dict["metrics"] else ["metrics/visits"]
        
        if not isinstance(result_dict["dimensions"], list):
            result_dict["dimensions"] = [result_dict["dimensions"]] if result_dict["dimensions"] else ["variables/page"]
        
        # Limit dimensions to 2
        if len(result_dict["dimensions"]) > 2:
            result_dict["dimensions"] = result_dict["dimensions"][:2]
            logger.info("Limited dimensions to 2 for optimal performance")
        
        # Ensure comparison_period exists
        if "comparison_period" not in result_dict:
            result_dict["comparison_period"] = ""
        
        # Ensure additional_context exists
        if "additional_context" not in result_dict:
            result_dict["additional_context"] = ""
        
        return result_dict

    async def _get_analytics_data_mcp(self, enhanced_query: AnalyticsQuery) -> Dict[str, Any]:
        """Stage 2: Retrieve data via MCP servers"""
        try:
            # Get current date from MCP server
            date_result = await self.mcp_client.get_current_date()
            if not date_result.get("success"):
                logger.warning("Failed to get current date from MCP server")
                current_date = "2024-11-20"  # Fallback
            else:
                current_date = date_result.get("date", "2024-11-20")
            
            # Validate schema via MCP
            validation_result = await self.mcp_client.validate_schema(
                enhanced_query.metrics,
                enhanced_query.dimensions
            )
            
            if not validation_result.get("success"):
                logger.warning(f"Schema validation failed: {validation_result.get('error')}")
            
            # Handle comparison queries
            if enhanced_query.comparison_period:
                primary_start, primary_end = self._parse_time_period(enhanced_query.time_period, current_date)
                comparison_start, comparison_end = self._parse_time_period(enhanced_query.comparison_period, current_date)
                
                logger.info(f"MCP comparison query - Primary: {primary_start} to {primary_end}, Comparison: {comparison_start} to {comparison_end}")
                
                result = await self.mcp_client.get_comparison_report(
                    metrics=enhanced_query.metrics,
                    dimensions=enhanced_query.dimensions,
                    primary_start=primary_start,
                    primary_end=primary_end,
                    comparison_start=comparison_start,
                    comparison_end=comparison_end
                )
                
                return {
                    "analysis_type": "comparison",
                    "output_format": enhanced_query.output_format,
                    "mcp_result": result,
                    "query_params": {
                        "metrics": enhanced_query.metrics,
                        "dimensions": enhanced_query.dimensions,
                        "primary_period": {"start": primary_start, "end": primary_end},
                        "comparison_period": {"start": comparison_start, "end": comparison_end}
                    },
                    "validation": validation_result
                }
            else:
                # Single period query
                start_date, end_date = self._parse_time_period(enhanced_query.time_period, current_date)
                
                logger.info(f"MCP single period query: {start_date} to {end_date}")
                
                result = await self.mcp_client.get_analytics_report(
                    metrics=enhanced_query.metrics,
                    dimensions=enhanced_query.dimensions,
                    start_date=start_date,
                    end_date=end_date
                )
                
                return {
                    "analysis_type": "single_period",
                    "output_format": enhanced_query.output_format,
                    "mcp_result": result,
                    "query_params": {
                        "metrics": enhanced_query.metrics,
                        "dimensions": enhanced_query.dimensions,
                        "start_date": start_date,
                        "end_date": end_date
                    },
                    "validation": validation_result
                }
            
        except Exception as e:
            logger.error(f"Error retrieving analytics data via MCP: {str(e)}")
            return {
                "error": str(e),
                "analysis_type": "error",
                "output_format": enhanced_query.output_format if hasattr(enhanced_query, 'output_format') else "detailed"
            }

    async def _analyze_data(self, enhanced_query: AnalyticsQuery, raw_data: Dict[str, Any]) -> AnalyticsResult:
        """Stage 3: Analyze data and generate insights"""
        try:
            is_comparison = raw_data.get("analysis_type") == "comparison"
            output_format = raw_data.get("output_format", "detailed")
            
            # Prepare analysis context
            mcp_result = raw_data.get("mcp_result", {})
            
            if is_comparison:
                analysis_context = f"""Analysis Type: Comparison Analysis via MCP Server
Output Format Requested: {output_format}
MCP Server Response Success: {mcp_result.get('success', False)}
Query Parameters: {json.dumps(raw_data.get('query_params', {}), indent=2)}

Data Summary:
- Primary Period Data: {json.dumps(mcp_result.get('primary_period', {}).get('metadata', {}), indent=2) if mcp_result.get('success') else 'Error retrieving data'}
- Comparison Period Data: {json.dumps(mcp_result.get('comparison_period', {}).get('metadata', {}), indent=2) if mcp_result.get('success') else 'Error retrieving data'}

CRITICAL: User requested "{output_format}" format - adapt your response accordingly."""
            else:
                analysis_context = f"""Analysis Type: {raw_data.get('analysis_type', 'unknown')} via MCP Server
Output Format Requested: {output_format}
MCP Server Response Success: {mcp_result.get('success', False)}
Query Parameters: {json.dumps(raw_data.get('query_params', {}), indent=2)}

Data Summary: {json.dumps(mcp_result.get('metadata', {}), indent=2) if mcp_result.get('success') else 'Error retrieving data'}

CRITICAL: User requested "{output_format}" format - adapt your response accordingly."""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_analysis_system_prompt()),
                ("human", "Enhanced Query: {enhanced_query}\n\nAnalysis Context: {analysis_context}\n\nOriginal Intent: {intent}\n\nREQUESTED OUTPUT FORMAT: {output_format}")
            ])
            
            chain = prompt | self.llm
            response = await chain.ainvoke({
                "enhanced_query": enhanced_query.enhanced_query,
                "analysis_context": analysis_context,
                "intent": enhanced_query.intent,
                "output_format": output_format
            })
            
            content = response.content.strip()
            logger.info(f"Analysis response: {content}")
            
            result_dict = self._extract_json_from_response(content)
            result_dict["raw_data"] = raw_data
            return AnalyticsResult(**result_dict)
            
        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            return AnalyticsResult(
                summary="Analysis completed with limited insights due to processing error.",
                key_insights=["Data retrieved successfully via MCP server", "Further analysis recommended"],
                data_analysis="Raw data available but detailed analysis encountered an error.",
                recommendations=["Review data manually", "Check MCP server status", "Verify query parameters"],
                raw_data=raw_data
            )

    def _extract_json_from_response(self, content: str) -> Dict:
        """Extract JSON from LLM response with multiple fallback methods"""
        # Method 1: Try direct JSON parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Method 2: Try to extract from markdown code blocks
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            if json_end > json_start:
                json_content = content[json_start:json_end].strip()
                try:
                    return json.loads(json_content)
                except json.JSONDecodeError:
                    pass
        
        # Method 3: Try to find any code block
        if "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            if json_end > json_start:
                json_content = content[json_start:json_end].strip()
                try:
                    return json.loads(json_content)
                except json.JSONDecodeError:
                    pass
        
        # Method 4: Try to find JSON-like content between braces
        start_brace = content.find("{")
        end_brace = content.rfind("}") + 1
        if start_brace != -1 and end_brace > start_brace:
            json_content = content[start_brace:end_brace]
            try:
                return json.loads(json_content)
            except json.JSONDecodeError:
                pass
        
        # Method 5: Fallback
        logger.warning("Could not extract JSON from response, using fallback")
        return {
            "summary": "Analysis completed with limited insights due to JSON parsing error.",
            "key_insights": ["Data processing encountered formatting issues", "Manual review recommended"],
            "data_analysis": "Response formatting error prevented detailed analysis.",
            "recommendations": ["Check system logs", "Retry analysis", "Contact support if issue persists"]
        }

    def _parse_time_period(self, time_period: str, current_date: str) -> tuple[str, str]:
        """Parse time period string into start and end dates"""
        from datetime import datetime, timedelta
        import calendar
        
        try:
            current = datetime.fromisoformat(current_date)
        except ValueError:
            logger.warning(f"Invalid current_date format: {current_date}, using today")
            current = datetime.now()
        
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
            date = current - timedelta(days=1)
            return date.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d")
        elif "last week" in time_period.lower() or "past week" in time_period.lower():
            end_date = current - timedelta(days=1)
            start_date = end_date - timedelta(days=6)
            return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        elif "last 30 days" in time_period.lower():
            end_date = current - timedelta(days=1)
            start_date = end_date - timedelta(days=29)
            return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        else:
            # Default to yesterday
            date = current - timedelta(days=1)
            return date.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d")