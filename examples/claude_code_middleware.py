"""Claude Code middleware: full conversation lifecycle with LCM.

Claude Code discovers MCP tools via .mcp.json. This example shows the
SYSTEM PROMPT you should add to instruct Claude Code to call lcm_ingest
and lcm_get_context automatically, completing the full read/write loop.

Setup:
  1. pip install lossless-agent
  2. Copy .mcp.json to your project root (see examples/claude_code_setup/)
  3. Add the SYSTEM_PROMPT below to your CLAUDE.md or system instructions

Since Claude Code is interactive and the agent itself decides which tools
to call, the middleware is implemented as instructions rather than code.
"""

# This system prompt instructs Claude Code to use the full LCM lifecycle.
# Add this to your project's CLAUDE.md file.
SYSTEM_PROMPT = """\
## Lossless Context Management (LCM)

You have access to LCM tools for persistent conversation memory.
Use them automatically in every session:

### At the START of each session:
Call `lcm_get_context` with your session_key to load prior context:
```
lcm_get_context(session_key="project-name", max_tokens=100000)
```
This returns summaries of past conversations + recent messages.
Use this context to understand what was discussed before.

### After EACH turn (user message + your response):
Call `lcm_ingest` to persist the conversation:
```
lcm_ingest(session_key="project-name", messages=[
  {"role": "user", "content": "<user's message>", "token_count": <estimate>},
  {"role": "assistant", "content": "<your response>", "token_count": <estimate>}
])
```

### When the session ENDS:
Call `lcm_session_end` to trigger final compaction:
```
lcm_session_end(session_key="project-name")
```

### For searching past context:
Use `lcm_grep`, `lcm_describe`, `lcm_expand` for targeted recall.

### Session key convention:
Use a stable identifier like the project name or directory: "my-project"
"""


def create_claude_md(project_name: str = "my-project") -> str:
    """Generate a CLAUDE.md file with LCM instructions.

    Write this to your project root as CLAUDE.md so Claude Code
    automatically picks up the LCM lifecycle instructions.
    """
    return f"""\
# Project Instructions

{SYSTEM_PROMPT.replace("project-name", project_name)}

## Recall Tools

- `lcm_grep(query)` — Search past messages and summaries
- `lcm_describe(summary_id)` — Get metadata for a summary node
- `lcm_expand(summary_id)` — Drill into a summary's source content
- `lcm_expand_query(conversation_id, query)` — AI-powered context search
- `lcm_stats()` — Database statistics
"""


if __name__ == "__main__":
    import sys
    project = sys.argv[1] if len(sys.argv) > 1 else "my-project"
    print(create_claude_md(project))
    print("\n# Save this as CLAUDE.md in your project root.")
