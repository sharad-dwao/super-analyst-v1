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
from .utils.mcp_pipeline import MCPAnalyticsPipeline

load_dotenv(".env.local")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI()

llm_model = "openai/gpt-4.1-mini"

client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

mcp_server_url = os.environ.get("MCP_SERVER_URL", "http://localhost:8001")
mcp_pipeline = MCPAnalyticsPipeline(
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
    mcp_server_url=mcp_server_url,
    base_url="https://openrouter.ai/api/v1"
)


class Request(BaseModel):
    messages: List[ClientMessage]


def get_tools_def():
    """Build OpenAI function definitions for MCP integration"""
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
    """Process user query through MCP-based pipeline"""
    logger.info(f"Processing query with MCP: {user_query}")
    result = await mcp_pipeline.process_query(user_query)
    return result


async def get_mcp_server_status() -> dict:
    """Get MCP server status and available tools"""
    try:
        await mcp_pipeline.initialize()
        return {
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
    except Exception as e:
        logger.error(f"Failed to get MCP server status: {str(e)}")
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


def stream_text(messages: List[ChatCompletionMessageParam], protocol: str = "data"):
    """Internal generator to handle streaming tokens and function calls"""
    draft_tool_calls = []
    draft_tool_calls_index = -1

    stream = client.chat.completions.create(
        messages=messages,
        model="openai/gpt-4.1-mini",
        stream=True,
        tools=get_tools_def(),
    )

    for chunk in stream:
        for choice in chunk.choices:
            if choice.finish_reason == "stop":
                continue

            elif choice.finish_reason == "tool_calls":
                for call in draft_tool_calls:
                    logger.info(f"Calling MCP tool {call['name']} with args {call['arguments']}")
                    yield f"9:{{\"toolCallId\":\"{call['id']}\",\"toolName\":\"{call['name']}\",\"args\":{call['arguments']}}}\n"

                import asyncio
                for call in draft_tool_calls:
                    if call["name"] == "analyze_with_mcp":
                        result = asyncio.run(analyze_with_mcp_streaming(
                            call["arguments"], call["id"]
                        ))
                        yield f"a:{{\"toolCallId\":\"{call['id']}\",\"toolName\":\"{call['name']}\",\"args\":{call['arguments']},\"result\":{json.dumps(result)}}}\n"
                    else:
                        result = asyncio.run(available_tools[call["name"]](
                            **json.loads(call["arguments"])
                        ))
                        
                        logger.info(f"MCP tool {call['name']} result: {result}")
                        yield f"a:{{\"toolCallId\":\"{call['id']}\",\"toolName\":\"{call['name']}\",\"args\":{call['arguments']},\"result\":{json.dumps(result)}}}\n"

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

            else:
                text = choice.delta.content
                if text:
                    yield f"0:{json.dumps(text)}\n"

        if not chunk.choices:
            usage = chunk.usage
            yield (
                f"e:{{\"finishReason\":\"{('tool-calls' if draft_tool_calls else 'stop')}\","
                f'"usage":{{"promptTokens":{usage.prompt_tokens},"completionTokens":{usage.completion_tokens}}},'
                f'"isContinued":false}}\n'
            )


async def analyze_with_mcp_streaming(arguments_str: str, tool_call_id: str):
    """Process MCP analysis with streaming support"""
    try:
        arguments = json.loads(arguments_str)
        user_query = arguments.get("user_query", "")
        
        logger.info(f"Starting MCP streaming analysis for: {user_query}")
        
        enhanced_query_data = await mcp_pipeline._enhance_query(user_query)
        
        stage1_text = f"**Stage 1 Complete:** Enhanced query - {enhanced_query_data.enhanced_query}\n\n"
        yield_stream_text(stage1_text)
        
        stage2_text = "**Stage 2:** Retrieving analytics data via MCP server...\n\n"
        yield_stream_text(stage2_text)
        
        raw_data = await mcp_pipeline._get_analytics_data_mcp(enhanced_query_data)
        
        stage2_complete = "**Stage 2 Complete:** Data retrieved successfully from MCP server\n\n"
        yield_stream_text(stage2_complete)
        
        stage3_text = "**Stage 3:** Analyzing data and generating insights...\n\n"
        yield_stream_text(stage3_text)
        
        final_result = await mcp_pipeline._analyze_data(enhanced_query_data, raw_data)
        
        return {
            "stage_1_enhancement": enhanced_query_data.dict() if hasattr(enhanced_query_data, 'dict') else enhanced_query_data,
            "stage_2_raw_data": raw_data,
            "stage_3_analysis": final_result.dict() if hasattr(final_result, 'dict') else final_result,
            "success": True,
            "mcp_info": {
                "server_url": mcp_server_url,
                "tools_available": len(mcp_pipeline.available_tools),
                "pipeline_type": "mcp_based"
            }
        }
        
    except Exception as e:
        logger.error(f"Error in MCP streaming analysis: {str(e)}")
        error_text = f"**Error:** {str(e)}\n\n"
        yield_stream_text(error_text)
        return {
            "error": str(e),
            "success": False,
            "mcp_info": {
                "server_url": mcp_server_url,
                "pipeline_type": "mcp_based"
            }
        }


def yield_stream_text(text: str):
    """Helper function to yield streaming text"""
    pass


PROMPT = None
with open(r"api\sys_prompt.md") as infile:
    PROMPT = "".join(infile.readlines())


@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query("data")):
    """Endpoint to handle incoming chat messages and stream responses using MCP servers"""
    system_msg = {
        "role": "system",
        "content": f"""{PROMPT}

You now have access to a powerful MCP (Model Context Protocol) based analytics system:

## MCP Integration Benefits:
- **Modular Architecture**: Analytics tools are deployed as separate MCP servers
- **Independent Updates**: MCP servers can be updated without affecting the main application
- **Scalable**: Multiple MCP servers can handle different analytics functions
- **Reliable**: Fault-tolerant communication with MCP servers

## Available MCP Functions:
1. **analyze_with_mcp**: Comprehensive analytics analysis using MCP servers
2. **get_mcp_server_status**: Check MCP server health and available tools

## MCP Pipeline Stages:
1. **Stage 1 - Query Enhancement**: Clarifies and structures user queries
2. **Stage 2 - MCP Data Retrieval**: Fetches data via MCP servers
3. **Stage 3 - Analysis**: Generates insights and recommendations

When users ask analytics questions, use the `analyze_with_mcp` function which will:
- Process queries through MCP servers for better reliability
- Handle complex analytics operations independently
- Provide comprehensive insights with proper error handling
- Support real-time streaming of analysis progress

MCP Server URL: {mcp_server_url}""",
    }

    messages = request.messages
    openai_messages = convert_to_openai_messages(messages)
    
    openai_messages.insert(0, system_msg)
    
    response = StreamingResponse(stream_text(openai_messages, protocol))
    response.headers["x-vercel-ai-data-stream"] = "v1"
    return response


@app.get("/api/mcp/status")
async def get_mcp_status():
    """Endpoint to check MCP server status"""
    return await get_mcp_server_status()


@app.on_event("startup")
async def startup_event():
    """Initialize MCP pipeline on startup"""
    try:
        await mcp_pipeline.initialize()
        logger.info("MCP pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MCP pipeline: {str(e)}")