
import os
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient


def get_mcp_client():
    server_path = (
        Path(__file__).resolve().parent.parent / "mcp_server.py"
    )

    return MultiServerMCPClient({
        "pharma_forecasting": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(server_path)],
            "env": dict(os.environ),
        }
    })
