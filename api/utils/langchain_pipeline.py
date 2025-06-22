import json
import logging
from typing import Dict, Any, List, AsyncGenerator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import AsyncCallbackHandler
from pydantic import BaseModel, Field
from .tools import get_report_adobe_analytics, get_report_adobe_analytics_multi_dimension, get_current_date
from .adobe_analytics_schema import (
    get_all_metrics, 
    get_all_dimensions, 
    get_metrics_by_category, 
    get_dimensions_by_category,
    get_analysis_template,
    search_metrics,
    search_dimensions,
    ANALYSIS_TEMPLATES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class StreamingCallbackHandler(AsyncCallbackHandler):
    """Custom callback handler for streaming LLM responses"""
    
    def __init__(self):
        self.tokens = []
        
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated"""
        self.tokens.append(token)
        # In a real implementation, you would yield this token
        # For now, we'll collect them
        pass


class AnalyticsQuery(BaseModel):
    """Structured analytics query after enhancement"""
    enhanced_query: str = Field(description="Enhanced and clarified user query")
    intent: str = Field(description="Primary intent of the query (e.g., 'performance_analysis', 'trend_analysis', 'comparison')")
    metrics: List[str] = Field(description="List of relevant metrics to analyze")
    dimensions: List[str] = Field(description="List of relevant dimensions to break down by (max 2)")
    time_period: str = Field(description="Time period for analysis (e.g., 'yesterday', 'last_week', 'last_month')")
    comparison_period: str = Field(description="Optional comparison period for month-to-month or period comparisons", default="")
    output_format: str = Field(description="Preferred output format (e.g., 'summary', 'detailed', 'table', 'chart', 'brief', 'executive')", default="detailed")
    additional_context: str = Field(description="Any additional context or requirements")


class AnalyticsResult(BaseModel):
    """Final analytics result with insights"""
    summary: str = Field(description="Executive summary of findings")
    key_insights: List[str] = Field(description="List of key insights discovered")
    data_analysis: str = Field(description="Detailed analysis of the data")
    recommendations: List[str] = Field(description="Actionable recommendations based on findings")
    raw_data: Dict[str, Any] = Field(description="Raw data from Adobe Analytics")


class LangChainAnalyticsPipeline:
    def __init__(self, openai_api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.llm = ChatOpenAI(
            model="openai/gpt-4.1-mini",
            api_key=openai_api_key,
            base_url=base_url,
            temperature=0.1,
            streaming=True  # Enable streaming
        )
        
        # Create a streaming version for analysis
        self.streaming_llm = ChatOpenAI(
            model="openai/gpt-4.1-mini",
            api_key=openai_api_key,
            base_url=base_url,
            temperature=0.1,
            streaming=True
        )

    def _get_enhancement_system_prompt(self) -> str:
        # Get sample metrics and dimensions by category for better context
        traffic_metrics = get_metrics_by_category("traffic")[:10]
        conversion_metrics = get_metrics_by_category("conversion")[:8]
        page_dimensions = get_dimensions_by_category("page")[:8]
        marketing_dimensions = get_dimensions_by_category("marketing")[:6]
        technology_dimensions = get_dimensions_by_category("technology")[:8]
        
        # Create analysis templates string without template variables
        templates_str = ""
        for template_name, template_info in ANALYSIS_TEMPLATES.items():
            templates_str += f"- {template_name}: {template_info['description']}\n"
        
        return f"""You are an expert analytics query enhancer using a predefined Adobe Analytics schema. Your job is to take user queries about web analytics and enhance them for better analysis.

## Available Metrics by Category:

**Traffic Metrics**: {', '.join(traffic_metrics)}
**Conversion Metrics**: {', '.join(conversion_metrics)}
**Campaign Metrics**: metrics/campaigninstances, metrics/clickthroughs
**Engagement Metrics**: metrics/timespent, metrics/averagetimespentonsite, metrics/averagetimespentonpage

## Available Dimensions by Category:

**Page Dimensions**: {', '.join(page_dimensions)}
**Marketing Dimensions**: {', '.join(marketing_dimensions)}
**Technology Dimensions**: {', '.join(technology_dimensions)}
**Geographic Dimensions**: variables/geocountry, variables/georegion, variables/geocity
**Time Dimensions**: variables/daterangemonth, variables/daterangeweek, variables/daterangeday

## Analysis Templates Available:
{templates_str}

## OUTPUT FORMAT DETECTION:
Pay special attention to how the user wants the output formatted. Look for keywords like:
- "summary" / "summarize" / "brief" → output_format: "summary"
- "detailed" / "in detail" / "comprehensive" → output_format: "detailed"
- "table" / "tabular" / "in a table" → output_format: "table"
- "chart" / "graph" / "visual" → output_format: "chart"
- "executive summary" / "high-level" → output_format: "executive"
- "bullet points" / "list" → output_format: "list"
- "quick overview" / "at a glance" → output_format: "brief"

If no specific format is mentioned, default to "detailed" for comprehensive analysis.

## COMPARISON ANALYSIS HANDLING:
When users ask for comparisons between time periods (e.g., "compare this month vs last month", "month over month"):
- Set the primary time_period to the most recent period being compared
- Set comparison_period to the earlier period for comparison
- Use specific month names when possible (e.g., "november_2024", "october_2024")
- For month comparisons, use variables/daterangemonth as a dimension

## TIME PERIOD SPECIFICATIONS:
- For "this month" use: "current_month" or specific month like "november_2024"
- For "last month" use: "previous_month" or specific month like "october_2024"
- For "month over month" use both periods: time_period="current_month", comparison_period="previous_month"
- For specific months use format: "january_2024", "february_2024", etc.

## IMPORTANT NOTES ABOUT TIME DIMENSIONS:
- When using time dimensions like variables/daterangemonth, variables/daterangeweek, etc., the system will automatically handle the time granularity
- For time-based analysis, it's better to use a non-time dimension and specify the time period separately
- If you need to analyze trends over time, use variables/daterangeday, variables/daterangeweek, or variables/daterangemonth as the primary dimension
- Time dimensions work best as the primary dimension, not as a secondary breakdown dimension

Your task:
1. Understand the user's intent and clarify ambiguous requests
2. Identify the most relevant metrics and dimensions from the predefined schema
3. Determine appropriate time periods if not specified
4. **DETECT THE PREFERRED OUTPUT FORMAT** from user language and keywords
5. For comparisons, identify both primary and comparison periods with specific dates
6. Enhance the query with analytics best practices
7. Provide structured output for the analytics engine

Guidelines:
- ONLY use metrics and dimensions from the predefined schema above
- If time period is not specified, default to reasonable periods (yesterday for daily data, last week for trends)
- Choose metrics that directly answer the user's question
- Select up to 2 dimensions that provide meaningful breakdowns
- For cross-dimensional analysis, choose complementary dimensions (e.g., page + device, channel + geo)
- Clarify vague terms (e.g., "performance" could mean visits, conversions, revenue)
- Consider what additional context might be helpful
- For month-to-month comparisons, set appropriate time periods and use time dimensions
- **PRIORITIZE the user's preferred output format** - this is critical for user satisfaction

Multi-Dimensional Analysis Examples:
- "page performance by device" → dimensions: ["variables/page", "variables/mobiledevicetype"]
- "traffic sources by geography" → dimensions: ["variables/referrertype", "variables/geocountry"]
- "content performance by time" → dimensions: ["variables/daterangeday", "variables/page"]
- "compare this month vs last month" → time_period: "current_month", comparison_period: "previous_month", dimensions: ["variables/daterangemonth"]

Output Format Examples:
- "give me a quick summary of traffic" → output_format: "summary"
- "show me detailed analysis of conversions" → output_format: "detailed"
- "put the results in a table" → output_format: "table"
- "brief overview of performance" → output_format: "brief"

You must respond with a valid JSON object with these exact fields:
- enhanced_query: string
- intent: string
- metrics: array of strings (use exact IDs from schema)
- dimensions: array of strings (max 2 elements, use exact IDs from schema)
- time_period: string
- comparison_period: string (empty if no comparison needed)
- output_format: string (detected from user query or "detailed" as default)
- additional_context: string

Respond with valid JSON only, no additional text or formatting."""

    def _get_analysis_system_prompt(self) -> str:
        return """You are an expert data analyst specializing in web analytics. Your job is to analyze Adobe Analytics data and provide actionable insights in the format preferred by the user.

## OUTPUT FORMAT GUIDELINES:

**CRITICAL: Always adapt your response format based on the user's preference:**

1. **"summary" or "brief"**: Provide concise, high-level findings in 2-3 sentences per section
2. **"detailed"**: Provide comprehensive analysis with full explanations (DEFAULT when not specified)
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

Your task:
1. **FIRST: Identify the requested output format and adapt accordingly**
2. Analyze the raw data from Adobe Analytics
3. Identify key patterns, trends, and anomalies
4. Provide clear, actionable insights in the preferred format
5. Make data-driven recommendations
6. Present findings in a business-friendly format

Guidelines for Multi-Dimensional Analysis:
- When analyzing cross-dimensional data, look for interaction patterns
- Identify which combinations perform best/worst
- Compare performance across different dimension values
- Look for opportunities in underperforming segments
- Consider seasonal or temporal patterns if time is a dimension

Guidelines for Comparison Analysis:
- When comparing time periods, calculate percentage changes and growth rates
- Identify significant trends and changes between periods
- Highlight both positive and negative changes
- Provide context for why changes might have occurred
- Focus on actionable insights from the comparison

Analysis Focus Areas:
- Performance leaders and laggards
- Unexpected patterns or anomalies
- Opportunities for optimization
- Segments requiring attention
- Cross-dimensional insights (e.g., mobile vs desktop performance by page)
- Time-based trends and comparisons
- Month-over-month or period-over-period changes

**REMEMBER: The output format preference is CRITICAL - users expect their specified format to be respected.**

**IMPORTANT FOR STREAMING: Structure your response to build progressively. Start with the summary, then add insights one by one, followed by detailed analysis, and finally recommendations. This creates a natural flow for real-time streaming.**

You must respond with a valid JSON object with these exact fields:
- summary: string (formatted according to user preference)
- key_insights: array of strings (formatted according to user preference)
- data_analysis: string (formatted according to user preference)
- recommendations: array of strings (formatted according to user preference)

Respond with valid JSON only, no additional text or formatting."""

    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query through the two-stage pipeline"""
        try:
            logger.info(f"Starting two-stage analysis for query: {user_query}")
            
            # Stage 1: Query Enhancement
            logger.info("Stage 1: Enhancing query...")
            enhanced_query_data = await self._enhance_query(user_query)
            
            # Stage 2: Data Retrieval (with comparison support)
            logger.info("Stage 2: Retrieving analytics data...")
            raw_data = await self._get_analytics_data(enhanced_query_data)
            
            # Stage 3: Analysis and Insights
            logger.info("Stage 3: Analyzing data and generating insights...")
            final_result = await self._analyze_data(enhanced_query_data, raw_data)
            
            return {
                "stage_1_enhancement": enhanced_query_data.dict() if hasattr(enhanced_query_data, 'dict') else enhanced_query_data,
                "stage_2_raw_data": raw_data,
                "stage_3_analysis": final_result.dict() if hasattr(final_result, 'dict') else final_result,
                "success": True,
                "schema_info": {
                    "total_metrics_available": len(get_all_metrics()),
                    "total_dimensions_available": len(get_all_dimensions()),
                    "schema_source": "predefined_adobe_analytics_schema"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }

    async def _enhance_query(self, user_query: str) -> AnalyticsQuery:
        """Stage 1: Enhance the user query using predefined schema"""
        try:
            # Create simple prompt template with only user_query variable
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_enhancement_system_prompt()),
                ("human", "User Query: {user_query}")
            ])
            
            # Create chain
            chain = prompt | self.llm
            
            # Get response
            response = await chain.ainvoke({"user_query": user_query})
            
            # Parse JSON response
            content = response.content.strip()
            logger.info(f"Enhancement response: {content}")
            
            # Extract and parse JSON
            result_dict = self._extract_json_from_response(content)
            
            # Validate that metrics and dimensions are from predefined schema
            validated_result = self._validate_schema_usage(result_dict)
            
            # Special handling for time dimensions
            validated_result = self._handle_time_dimensions(validated_result)
            
            return AnalyticsQuery(**validated_result)
                
        except Exception as e:
            logger.error(f"Error in query enhancement: {str(e)}")
            # Fallback to basic enhancement using predefined schema
            return AnalyticsQuery(
                enhanced_query=user_query,
                intent="general_analysis",
                metrics=["metrics/visits"],
                dimensions=["variables/page"],
                time_period="yesterday",
                comparison_period="",
                output_format="detailed",
                additional_context="Fallback enhancement due to processing error - using predefined schema defaults"
            )

    def _handle_time_dimensions(self, result_dict: Dict) -> Dict:
        """Special handling for time dimensions to avoid API issues"""
        dimensions = result_dict.get("dimensions", [])
        new_dimensions = []
        
        # Check if we have time dimensions and handle them appropriately
        for dim in dimensions:
            # If it's a time dimension
            if dim.startswith("variables/daterange"):
                # If it's the first dimension, keep it
                if len(new_dimensions) == 0:
                    new_dimensions.append(dim)
                    logger.info(f"Keeping time dimension as primary dimension: {dim}")
                else:
                    # If it's not the first dimension, replace with a non-time dimension
                    logger.warning(f"Replacing secondary time dimension {dim} with variables/page")
                    new_dimensions.append("variables/page")
            else:
                new_dimensions.append(dim)
        
        # Update dimensions in result
        result_dict["dimensions"] = new_dimensions
        
        # If we have a time dimension, make sure it's the primary dimension
        time_dims = [d for d in new_dimensions if d.startswith("variables/daterange")]
        if time_dims and new_dimensions[0] != time_dims[0]:
            # Swap to make time dimension the primary
            logger.info(f"Swapping dimensions to make time dimension primary: {time_dims[0]}")
            new_dimensions.remove(time_dims[0])
            new_dimensions.insert(0, time_dims[0])
            result_dict["dimensions"] = new_dimensions
            
        return result_dict

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
        
        # Method 5: Fallback - create basic structure
        logger.warning("Could not extract JSON from response, using fallback")
        return {
            "enhanced_query": "Basic query analysis",
            "intent": "general_analysis",
            "metrics": ["metrics/visits"],
            "dimensions": ["variables/page"],
            "time_period": "yesterday",
            "comparison_period": "",
            "output_format": "detailed",
            "additional_context": "JSON extraction failed, using defaults"
        }

    def _validate_schema_usage(self, result_dict: Dict) -> Dict:
        """Validate that the result uses metrics and dimensions from predefined schema"""
        all_metrics = get_all_metrics()
        all_dimensions = get_all_dimensions()
        
        # Validate metrics
        validated_metrics = []
        for metric in result_dict.get("metrics", []):
            if metric in all_metrics:
                validated_metrics.append(metric)
            else:
                # Try to find a close match
                matches = search_metrics(metric.replace("metrics/", ""))
                if matches:
                    validated_metrics.append(matches[0]["id"])
                    logger.info(f"Replaced metric {metric} with {matches[0]['id']}")
                else:
                    logger.warning(f"Metric {metric} not found in predefined schema")
        
        # Validate dimensions
        validated_dimensions = []
        for dimension in result_dict.get("dimensions", []):
            if dimension in all_dimensions:
                validated_dimensions.append(dimension)
            else:
                # Try to find a close match
                matches = search_dimensions(dimension.replace("variables/", ""))
                if matches:
                    validated_dimensions.append(matches[0]["id"])
                    logger.info(f"Replaced dimension {dimension} with {matches[0]['id']}")
                else:
                    logger.warning(f"Dimension {dimension} not found in predefined schema")
        
        # Update result with validated values
        result_dict["metrics"] = validated_metrics if validated_metrics else ["metrics/visits"]
        result_dict["dimensions"] = validated_dimensions if validated_dimensions else ["variables/page"]
        
        # Ensure output_format is set
        if "output_format" not in result_dict:
            result_dict["output_format"] = "detailed"
        
        return result_dict

    async def _get_analytics_data(self, enhanced_query: AnalyticsQuery) -> Dict[str, Any]:
        """Stage 2: Retrieve data from Adobe Analytics with support for multiple dimensions and comparisons"""
        try:
            # Convert time period to actual dates
            current_date = get_current_date()["date"]
            
            # Handle comparison queries
            if enhanced_query.comparison_period:
                # Get data for both periods
                primary_start, primary_end = self._parse_time_period(enhanced_query.time_period, current_date)
                comparison_start, comparison_end = self._parse_time_period(enhanced_query.comparison_period, current_date)
                
                logger.info(f"Comparison query - Primary: {primary_start} to {primary_end}, Comparison: {comparison_start} to {comparison_end}")
                
                # Get metrics and dimensions
                metrics = enhanced_query.metrics if enhanced_query.metrics else ["metrics/visits"]
                dimensions = enhanced_query.dimensions if enhanced_query.dimensions else ["variables/page"]
                
                # Limit to 2 dimensions for Adobe Analytics API
                if len(dimensions) > 2:
                    logger.warning(f"More than 2 dimensions requested, using first 2: {dimensions[:2]}")
                    dimensions = dimensions[:2]
                
                # Get data for primary period
                primary_data = self._get_single_period_data(metrics, dimensions, primary_start, primary_end, "primary")
                
                # Get data for comparison period
                comparison_data = self._get_single_period_data(metrics, dimensions, comparison_start, comparison_end, "comparison")
                
                return {
                    "analysis_type": "comparison",
                    "output_format": enhanced_query.output_format,
                    "query_params": {
                        "metrics": metrics,
                        "dimensions": dimensions,
                        "primary_period": {"start": primary_start, "end": primary_end},
                        "comparison_period": {"start": comparison_start, "end": comparison_end}
                    },
                    "primary_data": primary_data,
                    "comparison_data": comparison_data,
                    "schema_validation": "predefined_schema_used"
                }
            else:
                # Single period query
                start_date, end_date = self._parse_time_period(enhanced_query.time_period, current_date)
                
                logger.info(f"Single period query: {start_date} to {end_date}")
                
                # Get metrics and dimensions
                metrics = enhanced_query.metrics if enhanced_query.metrics else ["metrics/visits"]
                dimensions = enhanced_query.dimensions if enhanced_query.dimensions else ["variables/page"]
                
                # Limit to 2 dimensions for Adobe Analytics API
                if len(dimensions) > 2:
                    logger.warning(f"More than 2 dimensions requested, using first 2: {dimensions[:2]}")
                    dimensions = dimensions[:2]
                
                analytics_data = self._get_single_period_data(metrics, dimensions, start_date, end_date, "single")
                
                return {
                    "analysis_type": "single_period",
                    "output_format": enhanced_query.output_format,
                    "query_params": {
                        "metrics": metrics,
                        "dimensions": dimensions,
                        "start_date": start_date,
                        "end_date": end_date
                    },
                    "analytics_response": analytics_data,
                    "schema_validation": "predefined_schema_used"
                }
            
        except Exception as e:
            logger.error(f"Error retrieving analytics data: {str(e)}")
            return {
                "error": str(e),
                "analysis_type": "error",
                "output_format": enhanced_query.output_format if hasattr(enhanced_query, 'output_format') else "detailed",
                "query_params": {
                    "metrics": enhanced_query.metrics if enhanced_query.metrics else ["unknown"],
                    "dimensions": enhanced_query.dimensions if enhanced_query.dimensions else ["unknown"]
                }
            }

    def _get_single_period_data(self, metrics: List[str], dimensions: List[str], start_date: str, end_date: str, period_type: str) -> Dict[str, Any]:
        """Get analytics data for a single time period"""
        try:
            # Check if we're using time dimensions and handle appropriately
            has_time_dimension = any(dim.startswith("variables/daterange") for dim in dimensions)
            
            # Choose appropriate function based on number of dimensions
            if len(dimensions) == 1:
                analytics_data = get_report_adobe_analytics(
                    metrics=metrics,
                    dimension=dimensions[0],
                    start_date=start_date,
                    end_date=end_date
                )
                analysis_type = "single_dimensional"
            else:  # 2 dimensions
                # If we have a time dimension, make sure it's the primary dimension
                if has_time_dimension:
                    time_dim = next(dim for dim in dimensions if dim.startswith("variables/daterange"))
                    other_dim = next(dim for dim in dimensions if dim != time_dim)
                    analytics_data = get_report_adobe_analytics(
                        metrics=metrics,
                        dimension=time_dim,  # Time dimension as primary
                        dimension2=other_dim,
                        start_date=start_date,
                        end_date=end_date
                    )
                else:
                    analytics_data = get_report_adobe_analytics(
                        metrics=metrics,
                        dimension=dimensions[0],
                        dimension2=dimensions[1],
                        start_date=start_date,
                        end_date=end_date
                    )
                analysis_type = "multi_dimensional"
            
            return {
                "period_type": period_type,
                "analysis_type": analysis_type,
                "date_range": {"start": start_date, "end": end_date},
                "has_time_dimension": has_time_dimension,
                "analytics_response": analytics_data
            }
            
        except Exception as e:
            logger.error(f"Error getting {period_type} period data: {str(e)}")
            return {
                "period_type": period_type,
                "error": str(e),
                "date_range": {"start": start_date, "end": end_date}
            }

    async def _analyze_data(self, enhanced_query: AnalyticsQuery, raw_data: Dict[str, Any]) -> AnalyticsResult:
        """Stage 3: Analyze data and generate insights"""
        try:
            # Determine analysis context
            is_comparison = raw_data.get("analysis_type") == "comparison"
            output_format = raw_data.get("output_format", "detailed")
            
            if is_comparison:
                analysis_context = f"""Analysis Type: Comparison Analysis
Output Format Requested: {output_format}
Primary Period: {raw_data.get('query_params', {}).get('primary_period', {})}
Comparison Period: {raw_data.get('query_params', {}).get('comparison_period', {})}
Metrics: {', '.join(raw_data.get('query_params', {}).get('metrics', []))}
Dimensions: {', '.join(raw_data.get('query_params', {}).get('dimensions', []))}
Schema Source: Predefined Adobe Analytics Schema

CRITICAL: User requested "{output_format}" format - adapt your response accordingly."""
            else:
                dimensions = raw_data.get("query_params", {}).get("dimensions", [])
                has_time_dimension = any(dim.startswith("variables/daterange") for dim in dimensions)
                
                analysis_context = f"""Analysis Type: {raw_data.get('analysis_type', 'unknown')}
Output Format Requested: {output_format}
Dimensions: {', '.join(dimensions)}
Has Time Dimension: {has_time_dimension}
Date Range: {raw_data.get('query_params', {}).get('start_date', '')} to {raw_data.get('query_params', {}).get('end_date', '')}
Schema Source: Predefined Adobe Analytics Schema

CRITICAL: User requested "{output_format}" format - adapt your response accordingly."""
            
            # Create simple prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_analysis_system_prompt()),
                ("human", "Enhanced Query: {enhanced_query}\n\nAnalysis Context: {analysis_context}\n\nRaw Data: {raw_data}\n\nOriginal Intent: {intent}\n\nREQUESTED OUTPUT FORMAT: {output_format}")
            ])
            
            # Create chain
            chain = prompt | self.llm
            
            # Get response
            response = await chain.ainvoke({
                "enhanced_query": enhanced_query.enhanced_query,
                "analysis_context": analysis_context,
                "raw_data": json.dumps(raw_data, indent=2),
                "intent": enhanced_query.intent,
                "output_format": output_format
            })
            
            # Parse JSON response
            content = response.content.strip()
            logger.info(f"Analysis response: {content}")
            
            # Extract and parse JSON
            result_dict = self._extract_json_from_response(content)
            result_dict["raw_data"] = raw_data  # Add raw data
            return AnalyticsResult(**result_dict)
            
        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            # Fallback analysis
            return AnalyticsResult(
                summary="Analysis completed with limited insights due to processing error.",
                key_insights=["Data retrieved successfully using predefined schema", "Further analysis recommended"],
                data_analysis="Raw data available but detailed analysis encountered an error.",
                recommendations=["Review data manually", "Check query parameters", "Verify schema compatibility"],
                raw_data=raw_data
            )

    async def _analyze_data_streaming(self, enhanced_query: AnalyticsQuery, raw_data: Dict[str, Any]) -> AnalyticsResult:
        """Stage 3: Analyze data and generate insights with streaming support"""
        try:
            # This is a simplified version for now
            # In a full implementation, you would use streaming callbacks
            # to yield tokens as they're generated
            
            # For now, we'll use the regular analysis method
            # but structure it to be more streaming-friendly
            result = await self._analyze_data(enhanced_query, raw_data)
            
            # In a real streaming implementation, you would:
            # 1. Create a streaming callback handler
            # 2. Stream the summary first
            # 3. Stream each insight as it's generated
            # 4. Stream the detailed analysis
            # 5. Stream recommendations one by one
            
            return result
            
        except Exception as e:
            logger.error(f"Error in streaming data analysis: {str(e)}")
            # Fallback analysis
            return AnalyticsResult(
                summary="Analysis completed with limited insights due to processing error.",
                key_insights=["Data retrieved successfully using predefined schema", "Further analysis recommended"],
                data_analysis="Raw data available but detailed analysis encountered an error.",
                recommendations=["Review data manually", "Check query parameters", "Verify schema compatibility"],
                raw_data=raw_data
            )

    def _parse_time_period(self, time_period: str, current_date: str) -> tuple[str, str]:
        """Parse time period string into start and end dates with improved month handling"""
        from datetime import datetime, timedelta
        import calendar
        
        current = datetime.fromisoformat(current_date)
        
        # Handle specific month formats (e.g., "november_2024", "october_2024")
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
                
                # Get first and last day of the month
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
            # Go to first day of current month, then subtract 1 day to get last month
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
        elif "last 90 days" in time_period.lower() or "last quarter" in time_period.lower():
            end_date = current - timedelta(days=1)
            start_date = end_date - timedelta(days=89)
            return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        elif "last year" in time_period.lower() or "past year" in time_period.lower():
            end_date = current - timedelta(days=1)
            start_date = end_date - timedelta(days=364)
            return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        else:
            # Default to yesterday
            date = current - timedelta(days=1)
            return date.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d")