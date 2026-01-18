# TabPFN MCP Server

An MCP server that gives AI assistants the ability to train and run [TabPFN](https://github.com/PriorLabs/TabPFN) models for tabular classification and regression tasks.

This server uses [tabpfn-client](https://github.com/PriorLabs/tabpfn-client) to call the TabPFN API; no model training or inference is performed locally.

> Please note that server-side, remote training and inference is performed when calling the TabPFN API using their client. This involves data transfer to Prior Labs' servers. Refer to [TabPFN's privacy policy](https://priorlabs.github.io/TabPFN/privacy_policy.html) for details. See this [note](https://github.com/PriorLabs/tabpfn-client?tab=readme-ov-file#-stable-release:~:text=This%20is%20a,your%20organization%E2%80%99s%20policies.) in the TabPFN client README regarding data privacy and compliance.


## What it does

This server exposes two tools to MCP-compatible AI assistants:

- **`train_and_predict_classification`** - Train a classifier and get predictions with probabilities
- **`train_and_predict_regression`** - Train a regressor and get predictions

Both tools accept CSV files, automatically detect headers and whether test data has labels, compute metrics when labels are available, and can write predictions to output files.

## Installation

```bash
# Clone and install
git clone https://github.com/yourusername/tabpfn-mcp.git
cd tabpfn-mcp
uv pip install -e .
```

## Authentication

TabPFN requires an API token. The server automatically triggers browser-based login on first use if no credentials are cached.

To use a token explicitly, set the `TABPFN_TOKEN` environment variable in your MCP config (see examples below).

## Setup

### VS Code Copilot

Add to `.vscode/mcp.json` in your workspace:

```json
{
  "servers": {
    "tabpfn-mcp": {
      "command": "uv",
      "args": ["run", "tabpfn-mcp"]
    }
  }
}
```

### Claude Code

```bash
claude mcp add tabpfn-mcp -- uv run tabpfn-mcp
```

### Claude Desktop

Add the following to your `claude_desktop_config.json` file. The path to this file can be found in Claude Desktop under Settings > Developer > Local MCP servers > Edit Config.

```json
{
  "mcpServers": {
    "tabpfn": {
      "command": "uv",
      "args": ["run", "tabpfn-mcp"]
    }
  }
}
```

Restart Claude Desktop after saving.

## Example prompts

Once configured, you can ask the AI assistant to analyze your data. You need to provide locations to two CSV files, one each for the training set and the test set. The last column in the training set needs to contain the target values (labels). In case the test set also contains labels, model performance metrics are computed in addition to generating predictions.

**Classification:**
```
Train a TabPFN classifier on data/train.csv and predict labels for data/test.csv.
Save predictions to output/predictions.csv.
```

**Regression:**
```
I have housing data in housing_train.csv with price as the target.
Predict prices for housing_test.csv using TabPFN and save to predicted_prices.csv.
```

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Set up pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest
```

## Disclaimer

This MCP server is a client wrapper only. All data processing, training, and inference is performed by Prior Labs' TabPFN API on their servers. As the developer of this MCP server, I assume no responsibility for how data is handled, processed, or stored by the TabPFN service. Users must review and accept [TabPFN's privacy policy](https://priorlabs.ai/privacy-policy) and terms of service.