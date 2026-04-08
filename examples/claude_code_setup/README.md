# Claude Code Setup (2 Steps)

Give Claude Code lossless context management in under 2 minutes.

## Step 1: Install

```bash
pip install lossless-agent
```

## Step 2: Copy `.mcp.json` to your project root

```bash
cp .mcp.json /path/to/your/project/.mcp.json
```

Or create `.mcp.json` in your project root with:

```json
{
  "mcpServers": {
    "lossless-agent": {
      "command": "lossless-agent-mcp",
      "args": [
        "--db-path",
        "./data/lcm.db"
      ]
    }
  }
}
```

That's it. Claude Code automatically discovers `.mcp.json` and connects to
the MCP server. You now have `lcm_grep`, `lcm_describe`, and `lcm_expand`
tools available in every session.

## Customization

**Change the database path** — edit `--db-path` to any location:

```json
"args": ["--db-path", "~/.lcm/my-project.db"]
```

**Per-project databases** — each project gets its own `.mcp.json` with a
separate `--db-path`, so context stays isolated.

## Verify

Start Claude Code in your project directory. Ask:

> "What LCM tools do you have?"

You should see `lcm_grep`, `lcm_describe`, and `lcm_expand` in the response.
