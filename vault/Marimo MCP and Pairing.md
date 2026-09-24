---
tags: [platform, mcp, claude-code]
source: https://docs.marimo.io/guides/editor_features/mcp/
---
# Marimo MCP and Pairing

Two separate mechanisms; both let Claude Code act on a live notebook.

## 1. marimo as an MCP server (local)
```bash
pip install "marimo[mcp]"          # or: uvx "marimo[mcp]" edit ...
marimo edit notebook.py --mcp --no-token
# endpoint: http://localhost:PORT/mcp/server
claude mcp add --transport http marimo http://localhost:PORT/mcp/server
```
- `--no-token` only for local dev. `--mcp-allow-remote` disables Host-header checks behind proxies.
- Exposes all marimo AI tools plus prompts `active_notebooks` and `errors_summary`.
- Cursor/VS Code: `{"mcpServers": {"marimo": {"url": "http://localhost:PORT/mcp/server"}}}`.

## 2. marimo pair (recommended for Claude Code, works with molab)
- Skill: `npx skills add marimo-team/marimo-pair` or as a Claude Code plugin: `/plugin marketplace add marimo-team/marimo-pair` then `/plugin install marimo-pair@marimo-pair`.
- Needs bash, curl, jq. Talks to a running marimo server; auto-discovers `--no-token` servers, or uses `MARIMO_TOKEN` for authenticated ones (molab).
- Scripts: `scripts/discover-servers.sh`, `scripts/execute-code.sh`.
- Gives the agent: read variables, scratchpad execution, run cells, add/remove cells, manipulate UI.
- Usage: `/marimo-pair pair with me on notebooks/molab_afro_health_qa.py`.
- **Molab:** open the notebook on molab → actions menu → "Pair with an agent" → follow the instructions (they give the server URL + token to export).

## 3. marimo as MCP client (inside the notebook chat panel)
`marimo.toml`: `[mcp] presets = ["marimo", "context7"]`. Not needed for our workflow.

## Our setup (see [[Session Log]] for what is actually installed)
- Local: `uvx "marimo[mcp]" edit notebooks/molab_afro_health_qa.py --mcp --no-token --port 2718`, then `claude mcp add --transport http marimo http://localhost:2718/mcp/server`.
- Remote: molab "Pair with an agent" + the marimo-pair skill.

Related: [[Molab Platform]], [[Molab Notebook Plan]].
