#!/usr/bin/env python3
"""Google ADK + Lossless Agent setup.

Connects Google ADK to lossless-agent's MCP server using McpToolset,
giving your ADK agent lcm_grep, lcm_describe, and lcm_expand tools.

Install:
    pip install lossless-agent google-adk

Run:
    python examples/google_adk_setup.py

Requires: google-adk >= 0.3.0
"""
import asyncio
import sys

try:
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
except ImportError:
    print(
        "ERROR: google-adk is not installed.\n"
        "Install it with:\n\n"
        "    pip install google-adk\n"
    )
    sys.exit(1)


DB_PATH = "./data/lcm.db"


async def main() -> None:
    # 1. Connect to lossless-agent MCP server
    mcp_tools, cleanup = await McpToolset.from_server(
        connection_params=StdioConnectionParams(
            command="lossless-agent-mcp",
            args=["--db-path", DB_PATH],
        )
    )

    print(f"Connected! Tools: {[t.name for t in mcp_tools]}")

    # 2. Create an ADK agent with LCM tools
    agent = Agent(
        model="gemini-2.0-flash",
        name="lcm_agent",
        instruction=(
            "You are a helpful assistant with lossless context management. "
            "Use lcm_grep to search past conversations, lcm_describe to "
            "inspect summaries, and lcm_expand to drill into details."
        ),
        tools=mcp_tools,
    )

    # 3. Run the agent
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="lcm_app", session_service=session_service)

    session = await session_service.create_session(
        app_name="lcm_app", user_id="user-1"
    )

    from google.adk.agents import UserContent  # noqa: E402

    response = runner.run(
        session_id=session.id,
        user_id="user-1",
        new_message=UserContent(text="Search my history for 'deployment strategy'"),
    )

    async for event in response:
        if hasattr(event, "text") and event.text:
            print(f"Agent: {event.text}")

    # 4. Cleanup MCP connection
    await cleanup()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
