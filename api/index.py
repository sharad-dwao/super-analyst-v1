import os
import json
import logging
from typing import List
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from openai import OpenAI
from .utils.prompt import ClientMessage, convert_to_openai_messages
from .utils.langchain_pipeline import LangChainAnalyticsPipeline
from .utils.tools import (
    get_report_adobe_analytics,
    get_current_date,
    METRICS,
    DIMENSIONS,
)

# Load environment variables
load_dotenv(".env.local")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

llm_model = "openai/gpt-4.1-mini"

# Initialize OpenAI client (via OpenRouter)
client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Initialize LangChain Pipeline
langchain_pipeline = LangChainAnalyticsPipeline(
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


class Request(BaseModel):
    messages: List[ClientMessage]


def get_tools_def():
    """
    Dynamically build the OpenAI function definitions with enums
    for metrics and dimensions based on current caches.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "analyze_with_langchain",
                "description": "Analyze user query using LangChain two-stage pipeline: enhance query then fetch Adobe Analytics data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {
                            "type": "string",
                            "description": "The user's analytics question or request",
                        }
                    },
                    "required": ["user_query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_date",
                "description": "Return the current server date in YYYY-MM-DD format",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]


async def analyze_with_langchain(user_query: str) -> dict:
    """
    Process user query through LangChain two-stage pipeline
    """
    logger.info(f"Processing query with LangChain: {user_query}")
    result = await langchain_pipeline.process_query(user_query)
    return result


# Map function names to implementations
available_tools = {
    "analyze_with_langchain": analyze_with_langchain,
    "get_current_date": get_current_date,
}


def stream_text(messages: List[ChatCompletionMessageParam], protocol: str = "data"):
    """
    Internal generator to handle streaming tokens and function calls.
    """
    draft_tool_calls = []
    draft_tool_calls_index = -1

    # Create a streaming chat completion with dynamic function schema
    stream = client.chat.completions.create(
        messages=messages,
        model="openai/gpt-4.1-mini",
        stream=True,
        tools=get_tools_def(),
    )

    for chunk in stream:
        for choice in chunk.choices:
            # When conversation ends
            if choice.finish_reason == "stop":
                continue

            # When all tool calls are ready to invoke
            elif choice.finish_reason == "tool_calls":
                # Execute all collected tool calls
                for call in draft_tool_calls:
                    logger.info(
                        f"Calling tool {call['name']} with args {call['arguments']}"
                    )
                    yield f"9:{{\"toolCallId\":\"{call['id']}\",\"toolName\":\"{call['name']}\",\"args\":{call['arguments']}}}\n"

                # Execute tools and yield results with streaming support
                import asyncio
                for call in draft_tool_calls:
                    if call["name"] == "analyze_with_langchain":
                        # Handle async function with streaming
                        result = asyncio.run(analyze_with_langchain_streaming(
                            call["arguments"], call["id"]
                        ))
                        # The streaming is handled inside the function
                        # Just yield the final result marker
                        yield f"a:{{\"toolCallId\":\"{call['id']}\",\"toolName\":\"{call['name']}\",\"args\":{call['arguments']},\"result\":{json.dumps(result)}}}\n"
                    else:
                        # Handle sync function
                        result = available_tools[call["name"]](
                            **json.loads(call["arguments"])
                        )
                        
                        logger.info(f"Tool {call['name']} result: {result}")
                        yield f"a:{{\"toolCallId\":\"{call['id']}\",\"toolName\":\"{call['name']}\",\"args\":{call['arguments']},\"result\":{json.dumps(result)}}}\n"

            # Collect tool call arguments
            elif choice.delta.tool_calls:
                for tc in choice.delta.tool_calls:
                    if tc.id is not None:
                        draft_tool_calls_index += 1
                        draft_tool_calls.append(
                            {"id": tc.id, "name": tc.function.name, "arguments": ""}
                        )
                    else:
                        draft_tool_calls[draft_tool_calls_index][
                            "arguments"
                        ] += tc.function.arguments

            # Regular message content
            else:
                text = choice.delta.content
                if text:
                    yield f"0:{json.dumps(text)}\n"

        # At end of chunk, if no more choices, emit usage
        if not chunk.choices:
            usage = chunk.usage
            yield (
                f"e:{{\"finishReason\":\"{('tool-calls' if draft_tool_calls else 'stop')}\","
                f'"usage":{{"promptTokens":{usage.prompt_tokens},"completionTokens":{usage.completion_tokens}}},'
                f'"isContinued":false}}\n'
            )


async def analyze_with_langchain_streaming(arguments_str: str, tool_call_id: str):
    """
    Process LangChain analysis with streaming support
    """
    try:
        arguments = json.loads(arguments_str)
        user_query = arguments.get("user_query", "")
        
        logger.info(f"Starting streaming analysis for: {user_query}")
        
        # Stage 1: Query Enhancement (quick, no streaming needed)
        enhanced_query_data = await langchain_pipeline._enhance_query(user_query)
        
        # Stream stage 1 completion
        stage1_text = f"**Stage 1 Complete:** Enhanced query - {enhanced_query_data.enhanced_query}\n\n"
        yield_stream_text(stage1_text)
        
        # Stage 2: Data Retrieval
        stage2_text = "**Stage 2:** Retrieving analytics data...\n\n"
        yield_stream_text(stage2_text)
        
        raw_data = await langchain_pipeline._get_analytics_data(enhanced_query_data)
        
        stage2_complete = "**Stage 2 Complete:** Data retrieved successfully\n\n"
        yield_stream_text(stage2_complete)
        
        # Stage 3: Analysis with streaming
        stage3_text = "**Stage 3:** Analyzing data and generating insights...\n\n"
        yield_stream_text(stage3_text)
        
        # Get the analysis result with streaming
        final_result = await langchain_pipeline._analyze_data_streaming(enhanced_query_data, raw_data)
        
        # Return the complete result
        return {
            "stage_1_enhancement": enhanced_query_data.dict() if hasattr(enhanced_query_data, 'dict') else enhanced_query_data,
            "stage_2_raw_data": raw_data,
            "stage_3_analysis": final_result.dict() if hasattr(final_result, 'dict') else final_result,
            "success": True,
            "schema_info": {
                "total_metrics_available": len(METRICS),
                "total_dimensions_available": len(DIMENSIONS),
                "schema_source": "predefined_adobe_analytics_schema"
            }
        }
        
    except Exception as e:
        logger.error(f"Error in streaming analysis: {str(e)}")
        error_text = f"**Error:** {str(e)}\n\n"
        yield_stream_text(error_text)
        return {
            "error": str(e),
            "success": False
        }


def yield_stream_text(text: str):
    """Helper function to yield streaming text"""
    import sys
    # This is a placeholder - in the actual implementation,
    # we'll need to modify the LangChain pipeline to support streaming
    pass


# Load system prompt
PROMPT = None
with open(r"api\sys_prompt.md") as infile:
    PROMPT = "".join(infile.readlines())


@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query("data")):
    """
    Endpoint to handle incoming chat messages and stream responses.
    """
    # System message for analytics context
    system_msg = {
        "role": "system",
        "content": f"""{PROMPT}

You now have access to a powerful two-stage LangChain pipeline for analytics queries:

1. **Stage 1 - Query Enhancement**: The system will automatically enhance and clarify user queries, identifying relevant metrics, dimensions, and time periods.

2. **Stage 2 - Data Analysis**: After retrieving data from Adobe Analytics, the system provides structured insights, key findings, and actionable recommendations.

When users ask analytics questions, use the `analyze_with_langchain` function which will:
- Enhance their query for better analysis
- Fetch relevant data from Adobe Analytics  
- Provide structured insights and recommendations
- Return both raw data and business-friendly analysis

The pipeline handles complex analytics queries and provides comprehensive insights beyond simple data retrieval.

IMPORTANT: The analysis results will be streamed in real-time to provide immediate feedback to users. Present the results in a clear, structured format that builds progressively.""",
    }

    messages = request.messages
    openai_messages = convert_to_openai_messages(messages)
    
    # Add system message at the beginning
    openai_messages.insert(0, system_msg)
    
    response = StreamingResponse(stream_text(openai_messages, protocol))
    response.headers["x-vercel-ai-data-stream"] = "v1"
    return response