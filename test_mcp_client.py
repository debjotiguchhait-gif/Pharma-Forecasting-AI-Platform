import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(Path("mcp_server.py").resolve())],
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()

            print("\nTESTING MCP FORECAST TOOL")

            result = await session.call_tool(
                "forecast_medicine",
                arguments={"product_code": "0212000B0"},
            )

            if result.isError:
                raise RuntimeError(result.content)

            for item in result.content:
                if item.type == "text":
                    print(item.text)

            print("MCP CONNECTION SUCCESSFUL")
            print("Available tools:")

            for tool in tools.tools:
                print("-", tool.name)

async def test_langchain_adapter():
    from orchestrator.mcp_client import get_mcp_client

    client = get_mcp_client()
    tools = await client.get_tools()

    print("\nLANGCHAIN MCP TOOLS")
    for tool in tools:
        print(tool.name)

if __name__ == "__main__":
    asyncio.run(test_langchain_adapter())
