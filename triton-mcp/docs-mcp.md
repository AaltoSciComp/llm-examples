SciComp Docs MCP lets Cursor, Codex, and other MCP clients search the live [Aalto SciComp documentation](https://scicomp.aalto.fi/). The agent gets ranked pages and excerpts with published URLs — not invented cluster lore.

## Who it’s for

Anyone using an AI coding agent with MCP support on `code.triton.aalto.fi` (the login node for coding agents).

## What you get

- One tool: `search_scicomp_docs`
- Results from the live docs corpus, with links under `https://scicomp.aalto.fi/`
- Optional `path` filter when you know the area (e.g. `triton/`, `aalto/`)

## Connect (client config)

Add this to your MCP client settings (Cursor example: MCP servers config):

```json
{
  "mcpServers": {
    "scicomp-docs": {
      "url": "https://docs.triton.aalto.fi/mcp/"
    }
  }
}
```

Restart or reload MCP if your client requires it, then confirm `search_scicomp_docs` appears in the tool list.

## When the tool is used

**You ask.** Pose a docs question in chat; the agent should call `search_scicomp_docs` and cite the returned `https://scicomp.aalto.fi/` URLs. For example:

- “How do I request a GPU on Triton?”
- “Where is the Aalto storage quota documentation?”
- “What is the policy for AI agents on Triton?”

**The agent asks on its own.** In agent / auto mode, if the agent needs Triton or SciComp facts (partitions, modules, quotas, Slurm, policy, …), it may call the same tool without you mentioning docs — then fetch relevant sections instead of inventing cluster details.
