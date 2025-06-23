import os
import json
import logging
import asyncio
from datetime import datetime
from typing import List
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from .utils.prompt import ClientMessage, convert_to_openai_messages
from .utils.mcp_pipeline import MCPAnalyticsPipeline

load_dotenv(".env.local")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI()

llm_model = "openai/gpt-4.1-mini"

client = AsyncOpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    timeout=60.0,
)

mcp_server_url = os.environ.get("MCP_SERVER_URL", "http://localhost:8001")
mcp_pipeline = MCPAnalyticsPipeline(
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
    mcp_server_url=mcp_server_url,
    base_url="https://openrouter.ai/api/v1"
)

class Request(BaseModel):
    messages: List[ClientMessage]

def get_current_datetime_info():
    """Get current date and time information for LLM context"""
    now = datetime.now()
    return {
        "current_date": now.strftime("%Y-%m-%d"),
        "current_datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "current_year": now.year,
        "current_month": now.strftime("%B"),
        "current_month_num": now.month,
        "current_day": now.day,
        "day_of_week": now.strftime("%A"),
        "iso_date": now.isoformat()
    }

def get_tools_def():
    return [
        {
            "type": "function",
            "function": {
                "name": "analyze_with_mcp",
                "description": "Analyze user query using MCP servers for Adobe Analytics data",
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
                "name": "get_mcp_server_status",
                "description": "Check the status and available tools of MCP servers",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]

async def analyze_with_mcp(user_query: str) -> dict:
    logger.info(f"Starting MCP analysis for query: {user_query[:100]}...")
    try:
        result = await asyncio.wait_for(
            mcp_pipeline.process_query(user_query),
            timeout=120.0
        )
        logger.info("MCP analysis completed successfully")
        return result
    except asyncio.TimeoutError:
        logger.error("MCP analysis timed out")
        return {
            "error": "Analysis timed out",
            "success": False,
            "timeout": True
        }
    except Exception as e:
        logger.error(f"MCP analysis failed: {str(e)}")
        return {
            "error": str(e),
            "success": False
        }

async def get_mcp_server_status() -> dict:
    logger.info("Checking MCP server status")
    try:
        await mcp_pipeline.initialize()
        status = {
            "status": "connected",
            "server_url": mcp_server_url,
            "tools_available": len(mcp_pipeline.available_tools),
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description
                }
                for tool in mcp_pipeline.available_tools
            ]
        }
        logger.info(f"MCP server status: {status['status']}, tools: {status['tools_available']}")
        return status
    except Exception as e:
        logger.error(f"MCP server status check failed: {str(e)}")
        return {
            "status": "error",
            "server_url": mcp_server_url,
            "error": str(e),
            "tools_available": 0
        }

available_tools = {
    "analyze_with_mcp": analyze_with_mcp,
    "get_mcp_server_status": get_mcp_server_status,
}

async def stream_text(messages: List[ChatCompletionMessageParam], protocol: str = "data"):
    logger.info(f"Starting stream with {len(messages)} messages")
    draft_tool_calls = []
    draft_tool_calls_index = -1
    
    try:
        stream = await client.chat.completions.create(
            messages=messages,
            model=llm_model,
            stream=True,
            tools=get_tools_def(),
            timeout=60.0,
        )

        async for chunk in stream:
            try:
                for choice in chunk.choices:
                    if choice.finish_reason == "stop":
                        logger.debug("Stream finished normally")
                        continue

                    elif choice.finish_reason == "tool_calls":
                        logger.info(f"Processing {len(draft_tool_calls)} tool calls")
                        
                        for call in draft_tool_calls:
                            logger.info(f"Executing tool: {call['name']}")
                            yield f"9:{{\"toolCallId\":\"{call['id']}\",\"toolName\":\"{call['name']}\",\"args\":{call['arguments']}}}\n"

                        for call in draft_tool_calls:
                            try:
                                if call["name"] == "analyze_with_mcp":
                                    arguments = json.loads(call["arguments"])
                                    result = await analyze_with_mcp(arguments.get("user_query", ""))
                                else:
                                    arguments = json.loads(call["arguments"])
                                    result = await available_tools[call["name"]](**arguments)

                                logger.info(f"Tool {call['name']} completed successfully")
                                yield f"a:{{\"toolCallId\":\"{call['id']}\",\"toolName\":\"{call['name']}\",\"args\":{call['arguments']},\"result\":{json.dumps(result)}}}\n"

                            except Exception as tool_error:
                                logger.error(f"Tool {call['name']} failed: {str(tool_error)}")
                                error_result = {
                                    "error": str(tool_error),
                                    "success": False,
                                    "tool_name": call['name']
                                }
                                yield f"a:{{\"toolCallId\":\"{call['id']}\",\"toolName\":\"{call['name']}\",\"args\":{call['arguments']},\"result\":{json.dumps(error_result)}}}\n"

                    elif choice.delta.tool_calls:
                        for tc in choice.delta.tool_calls:
                            if tc.id is not None:
                                draft_tool_calls_index += 1
                                draft_tool_calls.append(
                                    {"id": tc.id, "name": tc.function.name, "arguments": ""}
                                )
                            else:
                                if draft_tool_calls_index >= 0:
                                    draft_tool_calls[draft_tool_calls_index]["arguments"] += tc.function.arguments

                    else:
                        text = choice.delta.content
                        if text:
                            yield f"0:{json.dumps(text)}\n"

            except Exception as chunk_error:
                logger.error(f"Error processing chunk: {str(chunk_error)}")
                continue

        if hasattr(chunk, 'usage') and chunk.usage:
            usage = chunk.usage
            yield (
                f"e:{{\"finishReason\":\"{('tool-calls' if draft_tool_calls else 'stop')}\","
                f'"usage":{{"promptTokens":{usage.prompt_tokens},"completionTokens":{usage.completion_tokens}}},'
                f'"isContinued":false}}\n'
            )
        else:
            yield f"e:{{\"finishReason\":\"{('tool-calls' if draft_tool_calls else 'stop')}\",\"usage\":{{\"promptTokens\":0,\"completionTokens\":0}},\"isContinued\":false}}\n"
            
    except Exception as stream_error:
        logger.error(f"Stream error: {str(stream_error)}")
        yield f"e:{{\"finishReason\":\"error\",\"error\":\"{str(stream_error)}\",\"isContinued\":false}}\n"

PROMPT = None
with open(r"api\sys_prompt.md") as infile:
    PROMPT = "".join(infile.readlines())

@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query("data")):
    logger.info(f"Received chat request with {len(request.messages)} messages")
    
    try:
        # Get current date/time information
        datetime_info = get_current_datetime_info()
        
        system_msg = {
            "role": "system",
            "content": f"""{PROMPT}

## CURRENT DATE AND TIME CONTEXT:
**CRITICAL: Always use this current date information for all time-based queries and calculations:**

- **Current Date**: {datetime_info['current_date']}
- **Current Year**: {datetime_info['current_year']}
- **Current Month**: {datetime_info['current_month']} ({datetime_info['current_month_num']})
- **Current Day**: {datetime_info['current_day']}
- **Day of Week**: {datetime_info['day_of_week']}
- **Full DateTime**: {datetime_info['current_datetime']}

**IMPORTANT TIME REFERENCE RULES:**
- When users say "this month", use {datetime_info['current_month']} {datetime_info['current_year']}
- When users say "last month", calculate the previous month from {datetime_info['current_date']}
- When users say "this year", use {datetime_info['current_year']}
- When users say "yesterday", calculate from {datetime_info['current_date']}
- When users say "last week", calculate the previous 7 days from {datetime_info['current_date']}
- Always ensure date calculations are relative to {datetime_info['current_date']}

## MCP Integration Benefits:
- **Modular Architecture**: Analytics tools are deployed as separate MCP servers
- **Independent Updates**: MCP servers can be updated without affecting the main application
- **Scalable**: Multiple MCP servers can handle different analytics functions
- **Reliable**: Fault-tolerant communication with MCP servers

## Available MCP Functions:
1. **analyze_with_mcp**: Comprehensive analytics analysis using MCP servers
2. **get_mcp_server_status**: Check MCP server health and available tools

## MCP Pipeline Stages:
1. **Stage 1 - Query Enhancement**: Clarifies and structures user queries with proper date context
2. **Stage 2 - MCP Data Retrieval**: Fetches data via MCP servers using correct date ranges
3. **Stage 3 - Analysis**: Generates insights and recommendations

When users ask analytics questions, use the `analyze_with_mcp` function which will:
- Process queries through MCP servers with accurate date context
- Handle complex analytics operations with proper time calculations
- Provide comprehensive insights with correct temporal references
- Support real-time streaming of analysis progress

**REMEMBER**: Always reference the current date context above when interpreting user queries about time periods.

MCP Server URL: {mcp_server_url}""",
        }

        messages = request.messages
        openai_messages = convert_to_openai_messages(messages)
        
        # Limit message history to prevent context overflow
        max_messages = 20
        if len(openai_messages) > max_messages:
            logger.info(f"Truncating message history from {len(openai_messages)} to {max_messages}")
            openai_messages = openai_messages[-max_messages:]
        
        openai_messages.insert(0, system_msg)
        
        logger.info(f"Processing {len(openai_messages)} messages with current date: {datetime_info['current_date']}")
        
        response = StreamingResponse(
            stream_text(openai_messages, protocol),
            media_type="text/plain"
        )
        response.headers["x-vercel-ai-data-stream"] = "v1"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        
        return response
        
    except Exception as e:
        logger.error(f"Chat handler error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mcp/status")
async def get_mcp_status():
    logger.info("MCP status endpoint called")
    return await get_mcp_server_status()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting FastAPI application")
    try:
        await mcp_pipeline.initialize()
        logger.info("MCP pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MCP pipeline: {str(e)}")
