from typing import Any, Dict, List

import gradio as gr
from core import MCPyLate

"""
MCPyLate Server
A Model Context Protocol server that provides search functionality using PyLate.
"""

mcpylate = MCPyLate()


def pylate_search_nfcorpus(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Search the PyLate with multi-vector models in the nfcorpus collection and return top-k hits
    Args:
        query: Search query string
        k: Number of results to return (default: 10)
        index_name: Name of index to search (default: use default index)
    Returns:
        List of search results with docid, score, text snippet, and index name
    """
    return mcpylate.search(query, k)


demo = gr.Interface(
    fn=pylate_search_nfcorpus,
    inputs=["text"],
    outputs="text",
    title="NFCorpus Search",
    description="Search in NFCorpus database index using PyLate",
)

demo.launch(mcp_server=True, share=True)
