import json
import logging
import os
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from .mcp_client import AdobeAnalyticsMCPClient, MCPTool

logger = logging.getLogger(__name__)


class AnalyticsQuery(BaseModel):
    enhanced_query: str = Field(description="Enhanced and clarified user query")
    intent: str = Field(description="Primary intent of the query")
    metrics: List[str] = Field(description="List of relevant metrics to analyze")
    dimensions: List[str] = Field(description="List of relevant dimensions")
    time_period: str = Field(description="Time period for analysis")
    comparison_period: str = Field(description="Optional comparison period", default="")
    output_format: str = Field(
        description="Preferred output format", default="detailed"
    )
    additional_context: str = Field(description="Additional context or requirements")


class AnalyticsResult(BaseModel):
    summary: str = Field(description="Executive summary of findings")
    key_insights: List[str] = Field(description="List of key insights")
    data_analysis: str = Field(description="Detailed analysis of the data")
    recommendations: List[str] = Field(description="Actionable recommendations")
    raw_data: Dict[str, Any] = Field(description="Raw data from analytics")


class MCPAnalyticsPipeline:
    def __init__(
        self,
        openai_api_key: str,
        mcp_server_url: str,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        # Fixed: Changed to correct model name
        self.llm = ChatOpenAI(
            model="openai/gpt-4.1-mini",
            api_key=openai_api_key,
            base_url=base_url,
            temperature=0.1,
            streaming=False,
            timeout=60.0,
            max_retries=2,
        )

        if not self._is_internal_url(mcp_server_url):
            raise ValueError(f"MCP server URL must be internal: {mcp_server_url}")

        self.mcp_client = AdobeAnalyticsMCPClient(mcp_server_url)
        self.available_tools = []
        self.mcp_server_url = mcp_server_url
        self._initialization_lock = asyncio.Lock()
        self._initialized = False

    def _is_internal_url(self, url: str) -> bool:
        internal_hosts = ["localhost", "127.0.0.1", "::1"]
        return any(host in url for host in internal_hosts)

    async def initialize(self):
        async with self._initialization_lock:
            if self._initialized:
                return

            try:
                logger.info("Initializing MCP pipeline")
                health = await asyncio.wait_for(
                    self.mcp_client.health_check(), timeout=10.0
                )

                if health.get("status") != "healthy":
                    logger.warning(f"MCP server health check failed: {health}")

                self.available_tools = await asyncio.wait_for(
                    self.mcp_client.list_available_tools(), timeout=15.0
                )

                logger.info(
                    f"MCP pipeline initialized with {len(self.available_tools)} tools"
                )
                self._initialized = True

                if not self.available_tools:
                    logger.warning("No tools discovered from MCP server")

            except asyncio.TimeoutError:
                logger.error("MCP pipeline initialization timed out")
                self.available_tools = []
            except Exception as e:
                logger.error(f"Failed to initialize MCP pipeline: {str(e)}")
                self.available_tools = []

    def _get_current_date_context(self) -> str:
        """Get current date context for LLM prompts"""
        now = datetime.now()
        return f"""
## CURRENT DATE CONTEXT (CRITICAL FOR TIME CALCULATIONS):
- **Today's Date**: {now.strftime("%Y-%m-%d")}
- **Current Year**: {now.year}
- **Current Month**: {now.strftime("%B")} ({now.month})
- **Current Day**: {now.day}
- **Day of Week**: {now.strftime("%A")}

**IMPORTANT**: Always use this current date when interpreting relative time periods like "this month", "last month", "yesterday", etc.
"""

    def _get_enhancement_system_prompt(self) -> str:
        tools_info = ""
        for tool in self.available_tools:
            tools_info += f"- {tool.name}: {tool.description}\n"

        if not tools_info:
            tools_info = "- get_analytics_report: Get Adobe Analytics report\n- get_comparison_report: Compare data between periods\n"

        current_date_context = self._get_current_date_context()

        return f"""You are an expert analytics query enhancer using MCP servers for Adobe Analytics. Your job is to take user queries about web analytics and enhance them for better analysis.

{current_date_context}

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

## TIME PERIOD HANDLING - CRITICAL:
**ALWAYS use the current date context above when interpreting time periods:**

For relative time periods, calculate based on the current date provided:
- "this month" → use current month and year from the date context
- "last month" → calculate previous month from current date
- "this year" → use current year from the date context
- "yesterday" → calculate from current date
- "last week" → calculate 7 days before current date
- "last 30 days" → calculate 30 days before current date

For specific months, use format: "month_year" (e.g., "november_2024", "october_2024")

## COMPARISON ANALYSIS HANDLING:
When users ask for comparisons between time periods:
- Set the primary time_period to the most recent period
- Set comparison_period to the earlier period for comparison
- Use specific month names when possible (e.g., "november_2024", "october_2024")

Your task:
1. Understand the user's intent and clarify ambiguous requests
2. **CRITICALLY IMPORTANT**: Use the current date context to properly interpret relative time periods
3. Identify the most relevant metrics and dimensions from the available schema
4. Determine appropriate time periods using the current date as reference
5. **DETECT THE PREFERRED OUTPUT FORMAT** from user language
6. For comparisons, identify both primary and comparison periods
7. Enhance the query with analytics best practices

Guidelines:
- Use only valid Adobe Analytics metrics and dimensions
- Limit dimensions to maximum 2 for optimal performance
- Choose metrics that directly answer the user's question
- For time-based analysis, consider using date range dimensions
- **ALWAYS reference the current date context when calculating time periods**

You must respond with a valid JSON object with these exact fields:
- enhanced_query: string
- intent: string
- metrics: array of strings (use valid Adobe Analytics metric IDs)
- dimensions: array of strings (max 2 elements, use valid dimension IDs)
- time_period: string (calculated using current date context)
- comparison_period: string (empty if no comparison needed)
- output_format: string (detected from user query or "detailed" as default)
- additional_context: string

Respond with valid JSON only, no additional text or formatting."""

    def _get_analysis_system_prompt(self) -> str:
        current_date_context = self._get_current_date_context()

        return f"""You are an expert data analyst specializing in web analytics using MCP servers. Your job is to analyze Adobe Analytics data and provide actionable insights in the format preferred by the user.

{current_date_context}

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
2. **Use the current date context to provide temporal context in your analysis**
3. Analyze the raw data from MCP servers
4. Identify key patterns, trends, and anomalies
5. Provide clear, actionable insights in the preferred format
6. Make data-driven recommendations
7. Present findings in a business-friendly format

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
        logger.info(f"Processing query: {user_query[:100]}...")

        try:
            if not self._initialized:
                await self.initialize()

            logger.info("Stage 1: Enhancing query")
            enhanced_query_data = await asyncio.wait_for(
                self._enhance_query(user_query), timeout=30.0
            )

            logger.info("Stage 2: Retrieving analytics data via MCP")
            raw_data = await asyncio.wait_for(
                self._get_analytics_data_mcp(enhanced_query_data), timeout=60.0
            )

            logger.info("Stage 3: Analyzing data and generating insights")
            final_result = await asyncio.wait_for(
                self._analyze_data(enhanced_query_data, raw_data), timeout=30.0
            )

            logger.info("Query processing completed successfully")
            return {
                "stage_1_enhancement": (
                    enhanced_query_data.dict()
                    if hasattr(enhanced_query_data, "dict")
                    else enhanced_query_data
                ),
                "stage_2_raw_data": raw_data,
                "stage_3_analysis": (
                    final_result.dict()
                    if hasattr(final_result, "dict")
                    else final_result
                ),
                "success": True,
                "mcp_info": {
                    "server_url": self.mcp_server_url,
                    "tools_available": len(self.available_tools),
                    "server_type": "mcp_adobe_analytics",
                    "pipeline_version": "mcp_v1.0",
                },
            }

        except asyncio.TimeoutError:
            logger.error("Query processing timed out")
            return {
                "error": "Query processing timed out",
                "success": False,
                "timeout": True,
                "mcp_info": {
                    "server_url": self.mcp_server_url,
                    "tools_available": len(self.available_tools),
                    "server_type": "mcp_adobe_analytics",
                },
            }
        except Exception as e:
            logger.error(f"Error in MCP pipeline: {str(e)}")
            return {
                "error": str(e),
                "success": False,
                "mcp_info": {
                    "server_url": self.mcp_server_url,
                    "tools_available": len(self.available_tools),
                    "server_type": "mcp_adobe_analytics",
                },
            }

    async def _enhance_query(self, user_query: str) -> AnalyticsQuery:
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self._get_enhancement_system_prompt()),
                    ("human", "User Query: {user_query}"),
                ]
            )

            chain = prompt | self.llm
            response = await chain.ainvoke({"user_query": user_query})

            content = response.content.strip()
            logger.debug(f"Enhancement response length: {len(content)}")

            result_dict = self._extract_json_from_response(content)
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
                additional_context="Fallback enhancement due to processing error",
            )

    def _validate_enhancement_result(self, result_dict: Dict) -> Dict:
        required_fields = [
            "enhanced_query",
            "intent",
            "metrics",
            "dimensions",
            "time_period",
            "output_format",
        ]
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

        if not isinstance(result_dict["metrics"], list):
            result_dict["metrics"] = (
                [result_dict["metrics"]]
                if result_dict["metrics"]
                else ["metrics/visits"]
            )

        if not isinstance(result_dict["dimensions"], list):
            result_dict["dimensions"] = (
                [result_dict["dimensions"]]
                if result_dict["dimensions"]
                else ["variables/page"]
            )

        if len(result_dict["dimensions"]) > 2:
            result_dict["dimensions"] = result_dict["dimensions"][:2]
            logger.info("Limited dimensions to 2 for optimal performance")

        if "comparison_period" not in result_dict:
            result_dict["comparison_period"] = ""

        if "additional_context" not in result_dict:
            result_dict["additional_context"] = ""

        return result_dict

    async def _get_analytics_data_mcp(
        self, enhanced_query: AnalyticsQuery
    ) -> Dict[str, Any]:
        try:
            date_result = await asyncio.wait_for(
                self.mcp_client.get_current_date(), timeout=10.0
            )

            if not date_result.get("success"):
                logger.warning("Failed to get current date from MCP server")
                current_date = datetime.now().strftime("%Y-%m-%d")
            else:
                current_date = date_result.get(
                    "date", datetime.now().strftime("%Y-%m-%d")
                )

            validation_result = await asyncio.wait_for(
                self.mcp_client.validate_schema(
                    enhanced_query.metrics, enhanced_query.dimensions
                ),
                timeout=15.0,
            )

            if not validation_result.get("success"):
                logger.warning(
                    f"Schema validation failed: {validation_result.get('error')}"
                )

            if enhanced_query.comparison_period:
                primary_start, primary_end = self._parse_time_period(
                    enhanced_query.time_period, current_date
                )
                comparison_start, comparison_end = self._parse_time_period(
                    enhanced_query.comparison_period, current_date
                )

                logger.info(
                    f"MCP comparison query - Primary: {primary_start} to {primary_end}, Comparison: {comparison_start} to {comparison_end}"
                )

                result = await asyncio.wait_for(
                    self.mcp_client.get_comparison_report(
                        metrics=enhanced_query.metrics,
                        dimensions=enhanced_query.dimensions,
                        primary_start=primary_start,
                        primary_end=primary_end,
                        comparison_start=comparison_start,
                        comparison_end=comparison_end,
                    ),
                    timeout=45.0,
                )

                return {
                    "analysis_type": "comparison",
                    "output_format": enhanced_query.output_format,
                    "mcp_result": result,
                    "query_params": {
                        "metrics": enhanced_query.metrics,
                        "dimensions": enhanced_query.dimensions,
                        "primary_period": {"start": primary_start, "end": primary_end},
                        "comparison_period": {
                            "start": comparison_start,
                            "end": comparison_end,
                        },
                    },
                    "validation": validation_result,
                    "current_date_used": current_date,
                }
            else:
                start_date, end_date = self._parse_time_period(
                    enhanced_query.time_period, current_date
                )

                logger.info(f"MCP single period query: {start_date} to {end_date}")

                result = await asyncio.wait_for(
                    self.mcp_client.get_analytics_report(
                        metrics=enhanced_query.metrics,
                        dimensions=enhanced_query.dimensions,
                        start_date=start_date,
                        end_date=end_date,
                    ),
                    timeout=45.0,
                )

                return {
                    "analysis_type": "single_period",
                    "output_format": enhanced_query.output_format,
                    "mcp_result": result,
                    "query_params": {
                        "metrics": enhanced_query.metrics,
                        "dimensions": enhanced_query.dimensions,
                        "start_date": start_date,
                        "end_date": end_date,
                    },
                    "validation": validation_result,
                    "current_date_used": current_date,
                }

        except asyncio.TimeoutError:
            logger.error("MCP data retrieval timed out")
            return {
                "error": "Data retrieval timed out",
                "analysis_type": "error",
                "output_format": (
                    enhanced_query.output_format
                    if hasattr(enhanced_query, "output_format")
                    else "detailed"
                ),
                "timeout": True,
            }
        except Exception as e:
            logger.error(f"Error retrieving analytics data via MCP: {str(e)}")
            return {
                "error": str(e),
                "analysis_type": "error",
                "output_format": (
                    enhanced_query.output_format
                    if hasattr(enhanced_query, "output_format")
                    else "detailed"
                ),
            }

    async def _analyze_data(
        self, enhanced_query: AnalyticsQuery, raw_data: Dict[str, Any]
    ) -> AnalyticsResult:
        try:
            is_comparison = raw_data.get("analysis_type") == "comparison"
            output_format = raw_data.get("output_format", "detailed")
            current_date_used = raw_data.get(
                "current_date_used", datetime.now().strftime("%Y-%m-%d")
            )

            mcp_result = raw_data.get("mcp_result", {})

            if is_comparison:
                analysis_context = f"""Analysis Type: Comparison Analysis via MCP Server
Output Format Requested: {output_format}
Current Date Reference: {current_date_used}
MCP Server Response Success: {mcp_result.get('success', False)}
Query Parameters: {json.dumps(raw_data.get('query_params', {}), indent=2)}

Data Summary:
- Primary Period Data: {json.dumps(mcp_result.get('primary_period', {}).get('metadata', {}), indent=2) if mcp_result.get('success') else 'Error retrieving data'}
- Comparison Period Data: {json.dumps(mcp_result.get('comparison_period', {}).get('metadata', {}), indent=2) if mcp_result.get('success') else 'Error retrieving data'}

CRITICAL: User requested "{output_format}" format - adapt your response accordingly."""
            else:
                analysis_context = f"""Analysis Type: {raw_data.get('analysis_type', 'unknown')} via MCP Server
Output Format Requested: {output_format}
Current Date Reference: {current_date_used}
MCP Server Response Success: {mcp_result.get('success', False)}
Query Parameters: {json.dumps(raw_data.get('query_params', {}), indent=2)}

Data Summary: {json.dumps(mcp_result.get('metadata', {}), indent=2) if mcp_result.get('success') else 'Error retrieving data'}

CRITICAL: User requested "{output_format}" format - adapt your response accordingly."""

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self._get_analysis_system_prompt()),
                    (
                        "human",
                        "Enhanced Query: {enhanced_query}\n\nAnalysis Context: {analysis_context}\n\nOriginal Intent: {intent}\n\nREQUESTED OUTPUT FORMAT: {output_format}",
                    ),
                ]
            )

            chain = prompt | self.llm
            response = await chain.ainvoke(
                {
                    "enhanced_query": enhanced_query.enhanced_query,
                    "analysis_context": analysis_context,
                    "intent": enhanced_query.intent,
                    "output_format": output_format,
                }
            )

            content = response.content.strip()
            logger.debug(f"Analysis response length: {len(content)}")

            result_dict = self._extract_json_from_response(content)
            result_dict["raw_data"] = raw_data
            return AnalyticsResult(**result_dict)

        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            return AnalyticsResult(
                summary="Analysis completed with limited insights due to processing error.",
                key_insights=[
                    "Data retrieved successfully via MCP server",
                    "Further analysis recommended",
                ],
                data_analysis="Raw data available but detailed analysis encountered an error.",
                recommendations=[
                    "Review data manually",
                    "Check MCP server status",
                    "Verify query parameters",
                ],
                raw_data=raw_data,
            )

    def _extract_json_from_response(self, content: str) -> Dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            if json_end > json_start:
                json_content = content[json_start:json_end].strip()
                try:
                    return json.loads(json_content)
                except json.JSONDecodeError:
                    pass

        if "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            if json_end > json_start:
                json_content = content[json_start:json_end].strip()
                try:
                    return json.loads(json_content)
                except json.JSONDecodeError:
                    pass

        start_brace = content.find("{")
        end_brace = content.rfind("}") + 1
        if start_brace != -1 and end_brace > start_brace:
            json_content = content[start_brace:end_brace]
            try:
                return json.loads(json_content)
            except json.JSONDecodeError:
                pass

        logger.warning("Could not extract JSON from response, using fallback")
        return {
            "summary": "Analysis completed with limited insights due to JSON parsing error.",
            "key_insights": [
                "Data processing encountered formatting issues",
                "Manual review recommended",
            ],
            "data_analysis": "Response formatting error prevented detailed analysis.",
            "recommendations": [
                "Check system logs",
                "Retry analysis",
                "Contact support if issue persists",
            ],
        }

    def _parse_time_period(
        self, time_period: str, current_date: str
    ) -> tuple[str, str]:
        from datetime import datetime, timedelta
        import calendar

        try:
            current = datetime.fromisoformat(current_date)
        except ValueError:
            logger.warning(f"Invalid current_date format: {current_date}, using today")
            current = datetime.now()

        if "_" in time_period and any(
            month in time_period.lower()
            for month in [
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            ]
        ):
            try:
                month_name, year = time_period.lower().split("_")
                year = int(year)
                month_num = {
                    "january": 1,
                    "february": 2,
                    "march": 3,
                    "april": 4,
                    "may": 5,
                    "june": 6,
                    "july": 7,
                    "august": 8,
                    "september": 9,
                    "october": 10,
                    "november": 11,
                    "december": 12,
                }[month_name]

                first_day = datetime(year, month_num, 1)
                last_day = datetime(
                    year, month_num, calendar.monthrange(year, month_num)[1]
                )

                return first_day.strftime("%Y-%m-%d"), last_day.strftime("%Y-%m-%d")
            except (ValueError, KeyError):
                logger.warning(f"Could not parse specific month format: {time_period}")

        if (
            "current_month" in time_period.lower()
            or "this month" in time_period.lower()
        ):
            first_day = current.replace(day=1)
            last_day = current.replace(
                day=calendar.monthrange(current.year, current.month)[1]
            )
            return first_day.strftime("%Y-%m-%d"), last_day.strftime("%Y-%m-%d")

        if (
            "previous_month" in time_period.lower()
            or "last month" in time_period.lower()
        ):
            first_of_current = current.replace(day=1)
            last_of_previous = first_of_current - timedelta(days=1)
            first_of_previous = last_of_previous.replace(day=1)

            return first_of_previous.strftime("%Y-%m-%d"), last_of_previous.strftime(
                "%Y-%m-%d"
            )

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
            date = current - timedelta(days=1)
            return date.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d")
