This folder contains example configs for several coding agents. Some files are meant to be copied into a real config path, while others are reference values to enter in a tool's settings UI.

## File map

### Codex

- `codex_config_aalto_azure.toml`: copy to `~/.codex/config.toml`

Set the required environment variable before starting Codex:
`MY_AZURE_KEY` 


### Cline for VS Code

- `cline_vscode_config_aalto_azure.txt`: reference values to paste into the Cline provider settings UI
- `cline_vscode_config_llm_gateway.txt`: reference values to paste into the Cline provider settings UI

These are not drop-in config files. Open Cline settings in VS Code, choose an OpenAI-compatible provider, and copy the values from the matching `.txt` file.

### Cursor

- `cursor_aalto_azure_config.txt`: reference values to enter in Cursor's model/provider settings UI

This is not a drop-in config file.

### OpenCode

- `opencode_config_aalto_azure.json`: copy to one of these locations:
	- Global, user-wide: `~/.config/opencode/opencode.json`
	- Per-project: `opencode.json` in your project root

Set `MY_AZURE_KEY` in your environment before use.

### pi

- `pi_config_llm_gateway_aalto_azure.json`: copy to `~/.pi/agents/models.json`

This file includes both an Aalto LLM Gateway provider and an Aalto Azure provider. Set the environment variables used by the provider you want to use:

- `AALTO_LLM_API_KEY` for the gateway provider
- `MY_AZURE_KEY` for the Azure provider