# MCPyLate

## Usage
MCPyLate is a boilerplate showing how to link PyLate to LLMs using MCP server.
The logic can be found in `core.py`, where you can edit the model as well as the dataset being indexed.
To use the server in Claude Code for example, use this config:
```
{
    "mcpServers": {
        "mcpylate": {
            "command": "uv",
            "args": [
                "--directory",
                "path_to_mcp_server.py",
                "run",
                "--extra",
                "eval",
                "mcp_server.py"
            ]
        }
    }
}
```
Since Claude Code has a strict startup delay on MCP server, we advise you to first launch the server manually using `uv run --directory path_to_mcp_server.py run --extra eval mcp_server.py`, it'll create the index and run the server. The server will then be faster to launch (because it'll only _load_ the elements) and should fit into the delay allowed.

The other solution is to simply use Gradio to create the MCP server and then use it as a remote MCP server. You can start it using `uv run --directory path_to_mcp_server.py run --extra eval gradio_mcp_server.py`
The corresponding config is then:
```
{
    "mcpServers": {
      "MCPyLate": {
        "command": "npx",
        "args": [
          "mcp-remote",
          "http://127.0.0.1:7860/gradio_api/mcp/sse"
        ]
      }
    }
  }
```
Eventually adapt the adress/port depending on your Gradio server.

## Context
Multi-vector search has shown very strong performance compared to single dense vector search in numerous domain, including out-of-domain, [long-context](https://x.com/antoine_chaffin/status/1919396926736257521) and [reasoning-intensive](https://x.com/antoine_chaffin/status/1925555110521798925) retrieval.

They are thus particularly well suited for modern retrieval use cases, including agentic workflows.
This MCP server is a demonstration of the use of PyLate models alongside its index optimized for multi-vector search, PLAID.

This tool has been built for the [Hugging Face MCP hackathon](https://huggingface.co/Agents-MCP-Hackathon), the corresponding space can be found [here](https://huggingface.co/spaces/Agents-MCP-Hackathon/MCPyLate).
The repository propose to search among the [leetcode split of the BRIGHT dataset](https://huggingface.co/datasets/xlangai/BRIGHT/viewer/documents/leetcode) using [Reason-ModernColBERT](https://huggingface.co/lightonai/Reason-ModernColBERT). This 150M parameters model outperforms 7B models on this reasoning-intensive retrieval benchmark, which requires reasoning to find relevant documents.
Although it can perform reasoning on its own, [it benefits from LLM reformulation](https://x.com/antoine_chaffin/status/1930625989643587598) and is thus particularly suited as a MCP to collaborate with an LLM.
Effectively, this demo allows to build a tool that can solve any leetcode problem by searching for problems that require similar algorithms and feed it to the LLM.

You can see it working in this example, where the agent find relevant leetcode solution, propose solutions with different complexity level as well as associated tests to validate them and even vizualisation tool.

https://youtu.be/Vvp2bPSfN68

If you want to use the space as a remote server, you can use this config:
```
{
    "mcpServers": {
      "MCPyLate": {
        "command": "npx",
        "args": [
          "mcp-remote",
          "https://agents-mcp-hackathon-mcpylate.hf.space/gradio_api/mcp/sse"
        ]
      }
    }
  }
```
The tool can easily be tweaked to index and search any dataset. For example, by indexing MS MARCO with [GTE-ModernColBERT](https://huggingface.co/lightonai/GTE-ModernColBERT-v1), this results in a Deep Research tool that can search among the web corpus and create reports based on successive queries.

https://youtu.be/WJx0XgYIlvA






