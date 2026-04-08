#!/usr/bin/env python3
"""Anthropic SDK + Lossless Agent MCP setup.

Connects the Anthropic Python SDK's MCP client to lossless-agent's
MCP server, giving your Claude-powered app lcm_grep, lcm_describe,
and lcm_expand tools via the standard tool_use flow.

Install:
    pip install lossless-agent anthropic[mcp]

Run:
    export ANTHROPIC_API_KEY=sk-...
    python examples/anthropic_agents_setup.py

Requires: anthropic >= 0.49.0 with MCP support
"""
import asyncio
import sys

try:
    import anthropic
    from anthropic.types.mcp import MCPServerStdio
except ImportError:
    print(
        "ERROR: anthropic with MCP support is not installed.\n"
        "Install it with:\n\n"
        "    pip install 'anthropic[mcp]'\n"
    )
    sys.exit(1)


DB_PATH = "./data/lcm.db"


async def main() -> None:
    # 1. Define the MCP server
    mcp_server = MCPServerStdio(
        command="lossless-agent-mcp",
        args=["--db-path", DB_PATH],
    )

    # 2. Create the Anthropic client with MCP
    client = anthropic.Anthropic()

    # 3. Use client.messages.create with MCP tools
    #    The SDK handles the MCP server lifecycle automatically
    async with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        mcp_servers=[mcp_server],
        messages=[
            {
                "role": "user",
                "content": "Search my conversation history for 'deployment strategy'",
            }
        ],
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)

    print()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
