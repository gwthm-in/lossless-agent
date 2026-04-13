"""Anthropic SDK middleware: full conversation lifecycle with LCM.

Shows a message loop wrapper that calls lcm_get_context before each API
call and lcm_ingest after, providing the full read/write lifecycle.

Requires: pip install lossless-agent anthropic
"""
from __future__ import annotations

from typing import Any, Optional


class LCMMiddleware:
    """Wraps Anthropic SDK calls with LCM ingestion and context assembly.

    This middleware manages the full lifecycle:
    1. Before each API call: loads prior context via lcm_get_context
    2. After each API call: persists messages via lcm_ingest
    3. On session end: runs final compaction via lcm_session_end

    Usage:
        import anthropic
        client = anthropic.Anthropic()
        lcm = LCMMiddleware(session_key="my-project")

        # Simple message loop
        while True:
            user_input = input("> ")
            if user_input == "quit":
                lcm.end_session()
                break
            response = lcm.chat(client, user_input)
            print(response)
    """

    def __init__(
        self,
        session_key: str,
        db_path: str = "./data/lcm.db",
        max_context_tokens: int = 100_000,
        model: str = "claude-sonnet-4-20250514",
        summarize_command: Optional[str] = None,
    ):
        self.session_key = session_key
        self.db_path = db_path
        self.max_context_tokens = max_context_tokens
        self.model = model
        self.messages: list[dict] = []
        self._mcp_cmd = ["lossless-agent-mcp", "--db-path", db_path]
        if summarize_command:
            self._mcp_cmd.extend(["--summarize-command", summarize_command])

    def _call_mcp_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call an MCP tool by invoking the server as a subprocess.

        For production use, maintain a persistent MCP connection instead.
        This simplified version uses the CLI directly for demonstration.
        """
        # In a real implementation, you'd use the MCP client protocol.
        # This example uses the lossless-agent Python API directly.
        from lossless_agent.store.database import Database
        from lossless_agent.store.conversation_store import ConversationStore
        from lossless_agent.store.message_store import MessageStore
        from lossless_agent.store.summary_store import SummaryStore
        from lossless_agent.engine.compaction import CompactionEngine
        from lossless_agent.engine.assembler import ContextAssembler, AssemblerConfig
        import asyncio

        db = Database(self.db_path)
        try:
            conv_store = ConversationStore(db)
            msg_store = MessageStore(db)
            sum_store = SummaryStore(db)

            if tool_name == "lcm_ingest":
                conv = conv_store.get_or_create(arguments["session_key"])
                ingested = []
                for msg in arguments["messages"]:
                    token_count = msg.get("token_count") or max(1, len(msg["content"]) // 4)
                    m = msg_store.append(
                        conversation_id=conv.id,
                        role=msg["role"],
                        content=msg["content"],
                        token_count=token_count,
                    )
                    ingested.append({"id": m.id, "seq": m.seq})
                return {"status": "ok", "messages_ingested": len(ingested)}

            elif tool_name == "lcm_get_context":
                conv = conv_store.get_or_create(arguments["session_key"])
                max_tokens = arguments.get("max_tokens", 100_000)
                assembler = ContextAssembler(
                    msg_store=msg_store,
                    sum_store=sum_store,
                    config=AssemblerConfig(max_context_tokens=max_tokens),
                )
                assembled = assembler.assemble(conv.id)
                context_text = assembler.format_context(assembled)
                return {
                    "status": "ok",
                    "context": context_text,
                    "total_tokens": assembled.total_tokens,
                }

            elif tool_name == "lcm_compact":
                conv = conv_store.get_or_create(arguments["session_key"])

                async def _truncation_summarize(prompt: str) -> str:
                    char_limit = 1200 * 4
                    if len(prompt) <= char_limit:
                        return prompt
                    return prompt[:char_limit] + "\n[Summary truncated]"

                engine = CompactionEngine(msg_store, sum_store, _truncation_summarize)
                created = asyncio.run(engine.compact_full_sweep(conv.id))
                return {"status": "ok", "summaries_created": len(created)}

            elif tool_name == "lcm_session_end":
                conv = conv_store.get_or_create(arguments["session_key"])

                async def _truncation_summarize(prompt: str) -> str:
                    char_limit = 1200 * 4
                    if len(prompt) <= char_limit:
                        return prompt
                    return prompt[:char_limit] + "\n[Summary truncated]"

                engine = CompactionEngine(msg_store, sum_store, _truncation_summarize)
                created = asyncio.run(engine.compact_full_sweep(conv.id))
                db.conn.execute(
                    "UPDATE conversations SET active = 0 WHERE id = ?", (conv.id,)
                )
                db.conn.commit()
                return {"status": "ok", "session_closed": True}

            else:
                return {"error": f"unknown tool: {tool_name}"}
        finally:
            db.close()

    def load_context(self) -> str:
        """Load prior conversation context from LCM."""
        result = self._call_mcp_tool("lcm_get_context", {
            "session_key": self.session_key,
            "max_tokens": self.max_context_tokens,
        })
        return result.get("context", "")

    def ingest(self, messages: list[dict]) -> dict:
        """Persist messages to LCM."""
        return self._call_mcp_tool("lcm_ingest", {
            "session_key": self.session_key,
            "messages": messages,
        })

    def end_session(self) -> dict:
        """Signal session end for final compaction."""
        return self._call_mcp_tool("lcm_session_end", {
            "session_key": self.session_key,
        })

    def chat(self, client: Any, user_message: str) -> str:
        """Complete chat turn with automatic LCM lifecycle.

        1. Load prior context
        2. Send message to Claude
        3. Ingest both messages into LCM
        4. Return assistant response
        """
        import anthropic

        # 1. Load context from LCM
        prior_context = self.load_context()

        # 2. Build messages with context
        system_msg = ""
        if prior_context:
            system_msg = f"[Prior conversation context from LCM]\n{prior_context}"

        self.messages.append({"role": "user", "content": user_message})

        # 3. Call Claude
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_msg if system_msg else anthropic.NOT_GIVEN,
            messages=self.messages,
        )
        assistant_text = response.content[0].text
        self.messages.append({"role": "assistant", "content": assistant_text})

        # 4. Ingest into LCM
        self.ingest([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_text},
        ])

        return assistant_text


# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------

def main():
    """Interactive chat with full LCM lifecycle."""
    import anthropic

    client = anthropic.Anthropic()
    lcm = LCMMiddleware(
        session_key="anthropic-sdk-demo",
        db_path="./data/lcm.db",
    )

    print("Chat with Claude (LCM-managed context). Type 'quit' to exit.")
    print("Prior context is automatically loaded from past sessions.\n")

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.strip().lower() in ("quit", "exit"):
            break

        response = lcm.chat(client, user_input)
        print(f"\nClaude: {response}\n")

    lcm.end_session()
    print("\nSession ended. Context saved for next time.")


if __name__ == "__main__":
    main()
