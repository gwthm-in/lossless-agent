"""Google ADK middleware: full conversation lifecycle with LCM.

Shows a complete ADK agent with before_agent_callback and
after_agent_callback that automatically handle context loading,
message ingestion, and compaction.

Requires: pip install lossless-agent google-adk
"""
from __future__ import annotations

import json
import asyncio
from typing import Any, Optional

# ADK imports (install with: pip install google-adk)
# from google.adk.agents import Agent
# from google.adk.runners import Runner
# from google.adk.sessions import InMemorySessionService
# from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams


# ------------------------------------------------------------------
# LCM Lifecycle Callbacks for ADK
# ------------------------------------------------------------------

class LCMLifecycleCallbacks:
    """ADK callback handler that manages the full LCM lifecycle.

    Attach this to your ADK agent to automatically:
    - Load context before each agent turn (before_agent_callback)
    - Ingest messages after each agent turn (after_agent_callback)

    Usage:
        lcm = LCMLifecycleCallbacks(session_key="my-project")

        agent = Agent(
            model="gemini-2.0-flash",
            name="my_agent",
            instruction=lcm.get_instruction(),
            tools=mcp_tools,
            before_agent_callback=lcm.before_agent,
            after_agent_callback=lcm.after_agent,
        )
    """

    def __init__(
        self,
        session_key: str,
        max_context_tokens: int = 100_000,
        db_path: str = "./data/lcm.db",
    ):
        self.session_key = session_key
        self.max_context_tokens = max_context_tokens
        self.db_path = db_path
        self._context_loaded = False

    def get_instruction(self) -> str:
        """Return agent instruction that includes LCM awareness."""
        return (
            f"You have lossless context management. Your session key is '{self.session_key}'. "
            "Past conversation context is automatically loaded before each turn. "
            "Use lcm_grep, lcm_describe, and lcm_expand for targeted recall of past discussions."
        )

    async def before_agent(self, callback_context: Any) -> Optional[Any]:
        """Before-agent callback: load context from LCM.

        In ADK, this is called before the agent processes each turn.
        We call lcm_get_context to retrieve prior conversation history.
        """
        # Access the MCP tools through the callback context
        # The agent's tools include lcm_get_context from the MCP server
        try:
            # Use the tool invocation mechanism provided by ADK
            # In practice, ADK's CallbackContext provides tool access
            tool_result = await callback_context.invoke_tool(
                "lcm_get_context",
                {
                    "session_key": self.session_key,
                    "max_tokens": self.max_context_tokens,
                },
            )
            if tool_result:
                result = json.loads(tool_result)
                if result.get("context"):
                    # Inject context into the conversation as a system message
                    callback_context.add_system_message(
                        f"[Prior conversation context]\n{result['context']}"
                    )
            self._context_loaded = True
        except Exception as e:
            # Don't fail the agent turn if context loading fails
            print(f"[LCM] Warning: failed to load context: {e}")

        return None  # Continue with normal agent processing

    async def after_agent(self, callback_context: Any) -> Optional[Any]:
        """After-agent callback: ingest messages into LCM.

        In ADK, this is called after the agent completes each turn.
        We call lcm_ingest to persist the conversation.
        """
        try:
            # Extract messages from the current turn
            messages = []
            for msg in callback_context.get_turn_messages():
                messages.append({
                    "role": msg.role,
                    "content": msg.content,
                    "token_count": len(msg.content) // 4,  # rough estimate
                })

            if messages:
                await callback_context.invoke_tool(
                    "lcm_ingest",
                    {
                        "session_key": self.session_key,
                        "messages": messages,
                    },
                )
        except Exception as e:
            print(f"[LCM] Warning: failed to ingest messages: {e}")

        return None

    async def on_session_end(self, callback_context: Any) -> None:
        """Call this when the session is ending for final compaction."""
        try:
            await callback_context.invoke_tool(
                "lcm_session_end",
                {"session_key": self.session_key},
            )
        except Exception as e:
            print(f"[LCM] Warning: failed to end session: {e}")


# ------------------------------------------------------------------
# Complete ADK setup example
# ------------------------------------------------------------------

async def main():
    """Complete example: ADK agent with full LCM lifecycle."""
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams

    # 1. Connect to LCM MCP server
    mcp_tools, cleanup = await McpToolset.from_server(
        connection_params=StdioConnectionParams(
            command="lossless-agent-mcp",
            args=["--db-path", "./data/lcm.db"],
        )
    )

    # 2. Set up LCM lifecycle callbacks
    lcm = LCMLifecycleCallbacks(session_key="my-adk-project")

    # 3. Create agent with LCM tools + lifecycle callbacks
    agent = Agent(
        model="gemini-2.0-flash",
        name="my_agent",
        instruction=lcm.get_instruction(),
        tools=mcp_tools,
        before_agent_callback=lcm.before_agent,
        after_agent_callback=lcm.after_agent,
    )

    # 4. Run the agent
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="lcm-demo", session_service=session_service)

    session = await session_service.create_session(app_name="lcm-demo", user_id="user-1")

    # Example conversation
    response = await runner.run(
        session_id=session.id,
        user_id="user-1",
        new_message="What were we working on last time?",
    )
    print(f"Agent: {response}")

    # 5. Clean up
    await lcm.on_session_end(runner)  # Final compaction
    await cleanup()  # Close MCP connection


if __name__ == "__main__":
    asyncio.run(main())
