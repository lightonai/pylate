"""
MCPyLate Server
A Model Context Protocol server that provides search functionality using PyLate.
"""

from typing import Any, Dict, List, Optional

from core import MCPyLate
from mcp.server.fastmcp import FastMCP


def register_tools(mcp: FastMCP, pylate: MCPyLate):
    """Register all tools with the MCP server."""

    @mcp.tool(
        name="pylate_search_nfcorpus",
        description="Perform a multi-vector search on the nfcorpus index. Returns top‑k hits with docid, score, and snippet.",
    )
    def pylate_search_nfcorpus(query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search the PyLate with multi-vector models and return top-k hits
        Args:
            query: Search query string
            k: Number of results to return (default: 10)
            index_name: Name of index to search (default: use default index)
        Returns:
            List of search results with docid, score, text snippet, and index name
        """
        return pylate.search(query, k)

    @mcp.tool(
        name="get_document",
        description="Retrieve a full document by its document ID from a Pyserini index.",
    )
    def get_document(
        docid: str, index_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve the full text of a document by its ID.

        Args:
            docid: Document ID to retrieve
            index_name: Name of index to search (default: use default index)

        Returns:
            Document with full text, or None if not found
        """
        return pylate.get_document(docid, index_name)


def main():
    """Main entry point for the server."""
    try:
        mcp = FastMCP("pylate-search-server")

        mcpylate = MCPyLate()
        register_tools(mcp, mcpylate)

        mcp.run(transport="stdio")

    except Exception as e:
        print(e)
        raise


if __name__ == "__main__":
    main()
