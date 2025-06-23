import json
import logging
from enum import Enum
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel
from typing import List, Optional, Any
from .attachment import ClientAttachment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ToolInvocationState(str, Enum):
    CALL = "call"
    PARTIAL_CALL = "partial-call"
    RESULT = "result"


class ToolInvocation(BaseModel):
    state: ToolInvocationState
    toolCallId: str
    toolName: str
    args: Any
    result: Any


class ClientMessage(BaseModel):
    role: str
    content: str
    experimental_attachments: Optional[List[ClientAttachment]] = None
    toolInvocations: Optional[List[ToolInvocation]] = None


async def prompt_enhance(client, llm_model, user_message):
    original_text = user_message.content
    enhance_prompt = [
        {
            "role": "system",
            "content": "You are a helpful assistant that improves user prompts for clarity and intent.",
        },
        {
            "role": "user",
            "content": "Here is the user prompt. Rewrite it to be clearer, more specific, and optimized for analytics tasks.",
        },
        {"role": "user", "content": original_text},
    ]

    response = await client.chat.completions.create(
        model=llm_model,
        messages=enhance_prompt,
        temperature=0.7,
    )

    enhanced_prompt = response.choices[0].message.content.strip()
    logger.info(enhanced_prompt)

    user_message.content = enhanced_prompt
    return user_message


def convert_to_openai_messages(
    messages: List[ClientMessage],
) -> List[ChatCompletionMessageParam]:
    openai_messages = []

    for message in messages:
        parts = []
        tool_calls = []

        parts.append({"type": "text", "text": message.content})

        if message.experimental_attachments:
            for attachment in message.experimental_attachments:
                if attachment.contentType.startswith("image"):
                    parts.append(
                        {"type": "image_url", "image_url": {"url": attachment.url}}
                    )

                elif attachment.contentType.startswith("text"):
                    parts.append({"type": "text", "text": attachment.url})

        # Only add tool calls if there are actual tool invocations with CALL state
        if message.toolInvocations:
            for toolInvocation in message.toolInvocations:
                if toolInvocation.state == ToolInvocationState.CALL:
                    tool_calls.append(
                        {
                            "id": toolInvocation.toolCallId,
                            "type": "function",
                            "function": {
                                "name": toolInvocation.toolName,
                                "arguments": json.dumps(toolInvocation.args),
                            },
                        }
                    )

        # Create the message with or without tool_calls
        message_dict = {
            "role": message.role,
            "content": parts,
        }
        
        # Only add tool_calls if there are actual tool calls
        if tool_calls:
            message_dict["tool_calls"] = tool_calls

        openai_messages.append(message_dict)

        # Only add tool result messages if there are actual tool invocations with RESULT state
        if message.toolInvocations:
            for toolInvocation in message.toolInvocations:
                if toolInvocation.state == ToolInvocationState.RESULT:
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": toolInvocation.toolCallId,
                        "content": json.dumps(toolInvocation.result),
                    }
                    openai_messages.append(tool_message)

    return openai_messages