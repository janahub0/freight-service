# the document search tool as an MCP server, so any MCP assistant (Claude Desktop, Cursor...) can call it.
# run it with: python mcp_server.py

from mcp.server.fastmcp import FastMCP
import agent

mcp = FastMCP("document-search")


@mcp.tool()
def search_documents(query: str, k: int = 3) -> str:
    """Search the notes on urban flooding and ESG reporting and return the best passages with their section names."""
    passages = agent.search(query, k)
    return "\n\n".join("[" + p["section"] + "] " + p["text"] for p in passages)


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression like '(40 - 10) / 40 * 100'."""
    return agent.calculator.invoke({"expression": expression})


if __name__ == "__main__":
    mcp.run()
