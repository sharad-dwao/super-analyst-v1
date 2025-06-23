import os
import sys
import types
import asyncio
import pytest
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")

    class AsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def aclose(self):
            pass

    class Timeout:
        def __init__(self, *args, **kwargs):
            pass

    class Limits:
        def __init__(self, *args, **kwargs):
            pass

    httpx_stub.AsyncClient = AsyncClient
    httpx_stub.Timeout = Timeout
    httpx_stub.Limits = Limits
    sys.modules["httpx"] = httpx_stub

if "pydantic" not in sys.modules:
    pydantic_stub = types.ModuleType("pydantic")

    class BaseModel:
        pass

    def Field(*args, **kwargs):
        return None

    pydantic_stub.BaseModel = BaseModel
    pydantic_stub.Field = Field
    sys.modules["pydantic"] = pydantic_stub

from api.utils.mcp_client import MCPClient


def test_is_internal_url_valid():
    client = MCPClient("http://localhost:8000")
    assert client._is_internal_url("http://localhost:1234")
    assert client._is_internal_url("https://127.0.0.1:5555/path")
    assert client._is_internal_url("http://[::1]/")


def test_is_internal_url_invalid():
    client = MCPClient("http://localhost:8000")
    assert not client._is_internal_url("http://example.com")
    assert not client._is_internal_url("http://10.0.0.1")
    assert not client._is_internal_url("not-a-url")


def test_async_client_verify_default():
    client = MCPClient("http://localhost:8000")
    async_client_mock = AsyncMock()
    async_client_mock.aclose = AsyncMock()

    async def run():
        with patch("api.utils.mcp_client.httpx.AsyncClient", return_value=async_client_mock) as mock_ac:
            async with client:
                pass
            kwargs = mock_ac.call_args.kwargs
            assert kwargs["verify"] is True

    asyncio.run(run())


def test_async_client_verify_custom_false():
    client = MCPClient("http://localhost:8000", verify=False)
    async_client_mock = AsyncMock()
    async_client_mock.aclose = AsyncMock()

    async def run():
        with patch("api.utils.mcp_client.httpx.AsyncClient", return_value=async_client_mock) as mock_ac:
            async with client:
                pass
            kwargs = mock_ac.call_args.kwargs
            assert kwargs["verify"] is False

    asyncio.run(run())

